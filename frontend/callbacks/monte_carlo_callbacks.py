"""
Monte Carlo Callbacks

This module provides callback functions for Monte Carlo simulation,
connecting user interactions to backend calculations and updating visualizations.

Requirements: 3.1, 3.5
"""

from dash import Output, Input, State
from dash.exceptions import PreventUpdate
import pandas as pd
import logging

from backend.calculators.monte_carlo_engine import (
    monte_carlo_simulation,
    calculate_percentile_bands,
    kelly_criterion_calculator,
    compare_risk_scenarios
)
from frontend.utils.error_handlers import (
    log_error
)

# Configure logging
logger = logging.getLogger(__name__)


def register_monte_carlo_callbacks(app):
    """
    Register all Monte Carlo-related callbacks.
    
    Parameters:
    -----------
    app : Dash app instance
        The Dash application to register callbacks with
    """
    
    @app.callback(
        Output('mc-risk-display', 'children'),
        Input('mc-risk-per-trade-slider', 'value')
    )
    def update_risk_display(risk_value):
        """Update risk percentage display when slider changes."""
        if risk_value is None:
            return "1.0%"
        return f"{risk_value:.1f}%"
    
    
    @app.callback(
        [
            Output('monte-carlo-results-store', 'data'),
            Output('mc-run-simulation-btn', 'children'),
            Output('mc-run-simulation-btn', 'disabled')
        ],
        Input('mc-run-simulation-btn', 'n_clicks'),
        [
            State('merged-data-store', 'data'),
            State('mc-n-simulations-input', 'value'),
            State('mc-initial-equity-input', 'value'),
            State('mc-risk-per-trade-slider', 'value')
        ],
        prevent_initial_call=True
    )
    def run_monte_carlo_simulation(n_clicks, merged_data, n_simulations, initial_equity, risk_per_trade):
        """
        Run Monte Carlo simulation when button is clicked.
        
        Validates parameters, calls backend simulation functions,
        and stores results for visualization callbacks.
        
        Validates: Requirements 3.1, 3.5
        """
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        
        try:
            # Parse merged data
            if not merged_data or 'data' not in merged_data:
                logger.warning("No data available for Monte Carlo simulation")
                return no_update, [
                    html.I(className="bi bi-play-fill me-2"),
                    "Jalankan Simulasi"
                ], False
            
            df = pd.read_json(merged_data['data'], orient='split')
            
            if df.empty:
                logger.warning("DataFrame is empty")
                return no_update, [
                    html.I(className="bi bi-play-fill me-2"),
                    "Jalankan Simulasi"
                ], False
            
            # Validate parameters
            if n_simulations is None or n_simulations < 100:
                logger.warning(f"Invalid n_simulations: {n_simulations}")
                n_simulations = 1000
            
            if initial_equity is None or initial_equity <= 0:
                logger.warning(f"Invalid initial_equity: {initial_equity}")
                initial_equity = 10000.0
            
            if risk_per_trade is None or not (0 < risk_per_trade <= 100):
                logger.warning(f"Invalid risk_per_trade: {risk_per_trade}")
                risk_per_trade = 1.0
            
            # Convert risk from percentage to decimal
            risk_decimal = risk_per_trade / 100.0
            
            # Update button to show loading state
            loading_button = [
                html.Span(className="spinner-border spinner-border-sm me-2"),
                "Menjalankan..."
            ]
            
            # Run Monte Carlo simulation
            logger.info(f"Running Monte Carlo: n={n_simulations}, equity=${initial_equity}, risk={risk_per_trade}%")
            mc_result = monte_carlo_simulation(
                df=df,
                n_simulations=int(n_simulations),
                initial_equity=float(initial_equity),
                risk_per_trade=risk_decimal,
                r_column='R_multiple'
            )
            
            # Calculate percentile bands for fan chart
            logger.info("Calculating percentile bands")
            percentile_bands = calculate_percentile_bands(
                equity_curves=mc_result['equity_curves'],
                percentiles=[5, 25, 50, 75, 95]
            )
            
            # Calculate Kelly Criterion
            logger.info("Calculating Kelly Criterion")
            kelly_result = kelly_criterion_calculator(df, r_column='R_multiple')
            
            # Compare risk scenarios
            logger.info("Comparing risk scenarios")
            risk_levels = [0.005, 0.01, 0.015, 0.02]  # 0.5%, 1%, 1.5%, 2%
            risk_comparison = compare_risk_scenarios(
                df=df,
                risk_levels=risk_levels,
                n_simulations=min(500, int(n_simulations)),  # Use fewer sims for comparison
                initial_equity=float(initial_equity),
                r_column='R_multiple'
            )
            
            # Prepare results for storage
            results = {
                'simulation_results': {
                    'final_equity_distribution': mc_result['final_equity_distribution'],
                    'max_drawdown_distribution': mc_result['max_drawdown_distribution'],
                    'prob_ruin': mc_result['prob_ruin'],
                    'prob_reach_target': mc_result['prob_reach_target'],
                    'median_final_equity': mc_result['median_final_equity'],
                    'percentile_5_equity': mc_result['percentile_5_equity'],
                    'percentile_95_equity': mc_result['percentile_95_equity'],
                    'percentile_95_dd': mc_result['percentile_95_dd'],
                    'n_simulations': mc_result['n_simulations'],
                    'initial_equity': mc_result['initial_equity'],
                    'risk_per_trade': mc_result['risk_per_trade']
                },
                'percentile_bands': percentile_bands,
                'kelly_criterion': kelly_result,
                'risk_comparison': risk_comparison.to_dict(orient='records'),
                'parameters': {
                    'n_simulations': int(n_simulations),
                    'initial_equity': float(initial_equity),
                    'risk_per_trade': risk_decimal,
                    'risk_percent': risk_per_trade
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info("Monte Carlo simulation completed successfully")
            
            # Reset button
            success_button = [
                html.I(className="bi bi-check-circle me-2"),
                "Simulasi Selesai"
            ]
            
            return results, success_button, False
            
        except ValueError as e:
            log_error(e, "Monte Carlo simulation validation")
            return no_update, [
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Error: " + str(e)
            ], False
            
        except Exception as e:
            log_error(e, "Monte Carlo simulation", include_traceback=True)
            return no_update, [
                html.I(className="bi bi-x-circle me-2"),
                "Error - Coba Lagi"
            ], False
    
    
    @app.callback(
        [
            Output('mc-n-simulations-input', 'value'),
            Output('mc-initial-equity-input', 'value'),
            Output('mc-risk-per-trade-slider', 'value')
        ],
        Input('mc-reset-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def reset_monte_carlo_parameters(n_clicks):
        """Reset Monte Carlo parameters to defaults."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        
        return 1000, 10000, 1.0
    
    
    @app.callback(
        Output('mc-summary-cards-container', 'children'),
        Input('monte-carlo-results-store', 'data')
    )
    def update_summary_cards(results):
        """
        Update summary cards with Monte Carlo results.
        
        Validates: Requirements 3.2
        """
        if not results or 'simulation_results' not in results:
            return html.Div()
        
        sim_results = results['simulation_results']
        initial_equity = sim_results['initial_equity']
        
        # Import component function
        from frontend.components.monte_carlo_viz import create_monte_carlo_summary_cards
        
        return create_monte_carlo_summary_cards()
    
    
    @app.callback(
        [
            Output('mc-median-equity-value', 'children'),
            Output('mc-p5-equity-value', 'children'),
            Output('mc-p95-equity-value', 'children'),
            Output('mc-prob-target-value', 'children')
        ],
        Input('monte-carlo-results-store', 'data')
    )
    def update_summary_card_values(results):
        """Update summary card values from Monte Carlo results."""
        if not results or 'simulation_results' not in results:
            return "-", "-", "-", "-"
        
        sim_results = results['simulation_results']
        
        median_equity = f"${sim_results['median_final_equity']:,.0f}"
        p5_equity = f"${sim_results['percentile_5_equity']:,.0f}"
        p95_equity = f"${sim_results['percentile_95_equity']:,.0f}"
        prob_target = f"{sim_results['prob_reach_target']:.1%}"
        
        return median_equity, p5_equity, p95_equity, prob_target
    
    
    @app.callback(
        Output('mc-equity-fan-chart', 'figure'),
        Input('monte-carlo-results-store', 'data')
    )
    def update_fan_chart(results):
        """
        Update equity curve fan chart with percentile bands.
        
        Validates: Requirements 3.2
        """
        import plotly.graph_objects as go
        
        if not results or 'percentile_bands' not in results:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Jalankan simulasi untuk melihat fan chart",
                xaxis_title="Jumlah Trades",
                yaxis_title="Equity ($)",
                template="plotly_white",
                height=450
            )
            return fig
        
        bands = results['percentile_bands']
        sim_results = results['simulation_results']
        initial_equity = sim_results['initial_equity']
        
        # Create time steps array
        time_steps = list(range(bands['time_steps']))
        
        # Create figure
        fig = go.Figure()
        
        # Add percentile bands as filled areas
        # 5th-95th percentile (outer band)
        fig.add_trace(go.Scatter(
            x=time_steps + time_steps[::-1],
            y=bands['p95'] + bands['p5'][::-1],
            fill='toself',
            fillcolor='rgba(0, 176, 80, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='5th-95th Percentile',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # 25th-75th percentile (inner band)
        fig.add_trace(go.Scatter(
            x=time_steps + time_steps[::-1],
            y=bands['p75'] + bands['p25'][::-1],
            fill='toself',
            fillcolor='rgba(0, 176, 80, 0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            name='25th-75th Percentile',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=bands['p50'],
            mode='lines',
            line=dict(color='darkgreen', width=3),
            name='Median (50th)',
            showlegend=True
        ))
        
        # Add initial equity reference line
        fig.add_hline(
            y=initial_equity,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial: ${initial_equity:,.0f}",
            annotation_position="right"
        )
        
        # Add 2x target line
        target_equity = initial_equity * 2
        fig.add_hline(
            y=target_equity,
            line_dash="dot",
            line_color="gold",
            annotation_text=f"Target 2x: ${target_equity:,.0f}",
            annotation_position="right"
        )
        
        # Update layout
        fig.update_layout(
            title=f"Equity Curve Fan Chart ({sim_results['n_simulations']} Simulations)",
            xaxis_title="Jumlah Trades",
            yaxis_title="Equity ($)",
            template="plotly_white",
            hovermode='x unified',
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    
    @app.callback(
        Output('mc-drawdown-histogram', 'figure'),
        Input('monte-carlo-results-store', 'data')
    )
    def update_drawdown_histogram(results):
        """
        Update drawdown distribution histogram.
        
        Validates: Requirements 3.3
        """
        import plotly.graph_objects as go
        import numpy as np
        
        if not results or 'simulation_results' not in results:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Jalankan simulasi untuk melihat distribusi drawdown",
                xaxis_title="Maximum Drawdown (%)",
                yaxis_title="Frequency",
                template="plotly_white",
                height=350
            )
            return fig
        
        sim_results = results['simulation_results']
        dd_distribution = sim_results['max_drawdown_distribution']
        percentile_95_dd = sim_results['percentile_95_dd']
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=dd_distribution,
            nbinsx=30,
            marker_color='lightcoral',
            name='Drawdown Distribution',
            hovertemplate='DD: %{x:.1f}%<br>Count: %{y}<extra></extra>'
        ))
        
        # Add 95th percentile line
        fig.add_vline(
            x=percentile_95_dd,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"95th: {percentile_95_dd:.1f}%",
            annotation_position="top right"
        )
        
        # Add mean line
        mean_dd = np.mean(dd_distribution)
        fig.add_vline(
            x=mean_dd,
            line_dash="dot",
            line_color="darkred",
            annotation_text=f"Mean: {mean_dd:.1f}%",
            annotation_position="top left"
        )
        
        # Update layout
        fig.update_layout(
            title="Distribusi Maximum Drawdown",
            xaxis_title="Maximum Drawdown (%)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=350,
            showlegend=False
        )
        
        return fig
    
    
    @app.callback(
        Output('mc-risk-comparison-table-container', 'children'),
        Input('monte-carlo-results-store', 'data')
    )
    def update_risk_comparison_table(results):
        """
        Update risk comparison table.
        
        Validates: Requirements 3.3
        """
        import dash_bootstrap_components as dbc
        from dash import html, dash_table
        
        if not results or 'risk_comparison' not in results:
            return dbc.Alert(
                "Jalankan simulasi untuk melihat perbandingan risk scenarios",
                color="light"
            )
        
        risk_comparison = pd.DataFrame(results['risk_comparison'])
        
        # Format the dataframe for display
        display_df = risk_comparison.copy()
        display_df['risk_percent'] = display_df['risk_percent'].apply(lambda x: f"{x:.2f}%")
        display_df['median_final_equity'] = display_df['median_final_equity'].apply(lambda x: f"${x:,.0f}")
        display_df['percentile_5_equity'] = display_df['percentile_5_equity'].apply(lambda x: f"${x:,.0f}")
        display_df['percentile_95_equity'] = display_df['percentile_95_equity'].apply(lambda x: f"${x:,.0f}")
        display_df['prob_ruin'] = display_df['prob_ruin'].apply(lambda x: f"{x:.1%}")
        display_df['prob_reach_target'] = display_df['prob_reach_target'].apply(lambda x: f"{x:.1%}")
        display_df['percentile_95_dd'] = display_df['percentile_95_dd'].apply(lambda x: f"{x:.1f}%")
        display_df['expected_return'] = display_df['expected_return'].apply(lambda x: f"{x:.1f}%")
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            'risk_percent': 'Risk/Trade',
            'median_final_equity': 'Median Equity',
            'percentile_5_equity': 'P5 Equity',
            'percentile_95_equity': 'P95 Equity',
            'prob_ruin': 'Prob Ruin',
            'prob_reach_target': 'Prob 2x',
            'percentile_95_dd': 'Max DD (95th)',
            'expected_return': 'Expected Return'
        })
        
        # Select columns to display
        display_columns = [
            'Risk/Trade', 'Median Equity', 'Prob Ruin', 
            'Max DD (95th)', 'Prob 2x', 'Expected Return'
        ]
        display_df = display_df[display_columns]
        
        # Create DataTable
        table = dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in display_df.columns],
            style_cell={
                'textAlign': 'center',
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
    
    
    @app.callback(
        Output('mc-kelly-criterion-display', 'children'),
        Input('monte-carlo-results-store', 'data')
    )
    def update_kelly_criterion_display(results):
        """
        Update Kelly Criterion display card.
        
        Validates: Requirements 3.5
        """
        import dash_bootstrap_components as dbc
        from dash import html
        import numpy as np
        
        if not results or 'kelly_criterion' not in results:
            return dbc.Alert(
                "Jalankan simulasi untuk melihat Kelly Criterion",
                color="light"
            )
        
        kelly = results['kelly_criterion']
        
        # Check if Kelly is valid
        if np.isnan(kelly['full_kelly']):
            return dbc.Alert(
                "Kelly Criterion tidak dapat dihitung (perlu data winners dan losers)",
                color="warning"
            )
        
        # Create display
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3(f"{kelly['full_kelly']:.2%}", className="text-primary mb-0"),
                        html.Small("Full Kelly", className="text-muted")
                    ], className="text-center")
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H3(f"{kelly['half_kelly']:.2%}", className="text-success mb-0"),
                        html.Small("Half Kelly", className="text-muted"),
                        html.Br(),
                        dbc.Badge("Recommended", color="success", className="mt-1")
                    ], className="text-center")
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.H3(f"{kelly['quarter_kelly']:.2%}", className="text-info mb-0"),
                        html.Small("Quarter Kelly", className="text-muted"),
                        html.Br(),
                        dbc.Badge("Conservative", color="info", className="mt-1")
                    ], className="text-center")
                ], md=4)
            ], className="mb-3"),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.Small([
                        html.Strong("Win Rate: "),
                        f"{kelly['win_rate']:.1%}"
                    ])
                ], md=4),
                dbc.Col([
                    html.Small([
                        html.Strong("Avg Win: "),
                        f"{kelly['avg_win']:.2f}R"
                    ])
                ], md=4),
                dbc.Col([
                    html.Small([
                        html.Strong("Avg Loss: "),
                        f"{kelly['avg_loss']:.2f}R"
                    ])
                ], md=4)
            ])
        ])
    
    
    @app.callback(
        Output('mc-risk-of-ruin-gauge', 'figure'),
        Input('monte-carlo-results-store', 'data')
    )
    def update_risk_of_ruin_gauge(results):
        """
        Update risk of ruin gauge.
        
        Validates: Requirements 3.3
        """
        import plotly.graph_objects as go
        
        if not results or 'simulation_results' not in results:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Jalankan simulasi untuk melihat risk of ruin",
                template="plotly_white",
                height=250
            )
            return fig
        
        sim_results = results['simulation_results']
        prob_ruin = sim_results['prob_ruin']
        
        # Determine color based on risk level
        if prob_ruin < 0.05:
            color = "green"
        elif prob_ruin < 0.15:
            color = "yellow"
        else:
            color = "red"
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_ruin * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability of Ruin (%)"},
            number={'suffix': "%", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 5], 'color': 'lightgreen'},
                    {'range': [5, 15], 'color': 'lightyellow'},
                    {'range': [15, 50], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 15
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    
    @app.callback(
        Output('mc-ruin-warning-message', 'children'),
        Input('monte-carlo-results-store', 'data')
    )
    def update_ruin_warning(results):
        """
        Display warning message if probability of ruin is high.
        
        Validates: Requirements 3.6, 10.3
        """
        import dash_bootstrap_components as dbc
        from dash import html
        
        if not results or 'simulation_results' not in results:
            return html.Div()
        
        sim_results = results['simulation_results']
        prob_ruin = sim_results['prob_ruin']
        risk_per_trade = sim_results['risk_per_trade']
        
        # Check if prob_ruin exceeds threshold
        if prob_ruin > 0.15:
            # Calculate recommended risk reduction
            recommended_risk = risk_per_trade * 0.5  # Reduce by 50%
            
            return dbc.Alert([
                html.H6([
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    "PERINGATAN: Risk Terlalu Tinggi!"
                ], className="alert-heading mb-2"),
                html.P([
                    f"Probability of ruin Anda adalah {prob_ruin:.1%}, yang terlalu tinggi! ",
                    "Ini berarti ada kemungkinan signifikan equity Anda akan turun lebih dari 50%."
                ], className="mb-2"),
                html.Hr(),
                html.P([
                    html.Strong("Rekomendasi: "),
                    f"Turunkan risk per trade dari {risk_per_trade:.1%} ke {recommended_risk:.1%} atau lebih rendah. ",
                    "Target probability of ruin < 10% untuk trading yang aman."
                ], className="mb-0")
            ], color="danger", className="mt-2")
        
        elif prob_ruin > 0.05:
            return dbc.Alert([
                html.H6([
                    html.I(className="bi bi-exclamation-circle-fill me-2"),
                    "Perhatian: Risk Moderate"
                ], className="alert-heading mb-2"),
                html.P([
                    f"Probability of ruin Anda adalah {prob_ruin:.1%}. ",
                    "Ini masih acceptable untuk experienced traders, tapi pertimbangkan untuk menurunkan risk jika Anda risk-averse."
                ], className="mb-0")
            ], color="warning", className="mt-2")
        
        else:
            return dbc.Alert([
                html.H6([
                    html.I(className="bi bi-check-circle-fill me-2"),
                    "Risk Level Aman"
                ], className="alert-heading mb-2"),
                html.P([
                    f"Probability of ruin Anda hanya {prob_ruin:.1%}. ",
                    "Risk level ini aman dan sustainable untuk long-term trading."
                ], className="mb-0")
            ], color="success", className="mt-2")
    
    
    @app.callback(
        Output('mc-insights-panel', 'children'),
        Input('monte-carlo-results-store', 'data')
    )
    def update_insights_panel(results):
        """
        Generate insights and recommendations from Monte Carlo results.
        
        Validates: Requirements 3.6, 10.3
        """
        import dash_bootstrap_components as dbc
        from dash import html
        import numpy as np
        
        if not results or 'simulation_results' not in results:
            return dbc.Alert(
                "Jalankan simulasi untuk melihat insights dan rekomendasi",
                color="light"
            )
        
        sim_results = results['simulation_results']
        kelly = results['kelly_criterion']
        
        prob_ruin = sim_results['prob_ruin']
        percentile_95_dd = sim_results['percentile_95_dd']
        median_equity = sim_results['median_final_equity']
        initial_equity = sim_results['initial_equity']
        risk_per_trade = sim_results['risk_per_trade']
        prob_reach_target = sim_results['prob_reach_target']
        
        insights = []
        
        # Insight 1: Overall risk assessment
        if prob_ruin < 0.05:
            insights.append(
                html.Li([
                    html.I(className="bi bi-check-circle-fill text-success me-2"),
                    html.Strong("Risk Aman: "),
                    f"Probability of ruin hanya {prob_ruin:.1%}. Risk {risk_per_trade:.1%} per trade aman untuk long-term trading."
                ])
            )
        elif prob_ruin < 0.15:
            insights.append(
                html.Li([
                    html.I(className="bi bi-exclamation-circle-fill text-warning me-2"),
                    html.Strong("Risk Moderate: "),
                    f"Probability of ruin {prob_ruin:.1%}. Pertimbangkan turunkan risk ke {risk_per_trade * 0.7:.1%} untuk lebih aman."
                ])
            )
        else:
            insights.append(
                html.Li([
                    html.I(className="bi bi-x-circle-fill text-danger me-2"),
                    html.Strong("Risk Tinggi: "),
                    f"BAHAYA! Probability of ruin {prob_ruin:.1%}. Turunkan risk segera ke {risk_per_trade * 0.5:.1%} atau kurang."
                ])
            )
        
        # Insight 2: Drawdown assessment
        if percentile_95_dd > 30:
            insights.append(
                html.Li([
                    html.I(className="bi bi-arrow-down-circle text-danger me-2"),
                    html.Strong("Drawdown Tinggi: "),
                    f"95th percentile drawdown adalah {percentile_95_dd:.1f}%. Ini berarti 5% worst scenarios mengalami DD > 30%. "
                    "Pertimbangkan turunkan risk atau improve strategy."
                ])
            )
        elif percentile_95_dd > 20:
            insights.append(
                html.Li([
                    html.I(className="bi bi-dash-circle text-warning me-2"),
                    html.Strong("Drawdown Moderate: "),
                    f"95th percentile drawdown adalah {percentile_95_dd:.1f}%. Masih acceptable, tapi monitor closely."
                ])
            )
        else:
            insights.append(
                html.Li([
                    html.I(className="bi bi-check-circle text-success me-2"),
                    html.Strong("Drawdown Terkontrol: "),
                    f"95th percentile drawdown hanya {percentile_95_dd:.1f}%. Excellent risk management!"
                ])
            )
        
        # Insight 3: Expected return
        expected_return_pct = ((median_equity - initial_equity) / initial_equity) * 100
        insights.append(
            html.Li([
                html.I(className="bi bi-graph-up text-primary me-2"),
                html.Strong("Expected Return: "),
                f"Median final equity ${median_equity:,.0f} ({expected_return_pct:+.1f}%). "
                f"Probability mencapai 2x initial equity: {prob_reach_target:.1%}."
            ])
        )
        
        # Insight 4: Kelly Criterion comparison
        if not np.isnan(kelly['full_kelly']):
            if risk_per_trade > kelly['full_kelly']:
                insights.append(
                    html.Li([
                        html.I(className="bi bi-exclamation-triangle text-warning me-2"),
                        html.Strong("Over-Kelly: "),
                        f"Risk Anda ({risk_per_trade:.1%}) melebihi Full Kelly ({kelly['full_kelly']:.1%}). "
                        f"Rekomendasi: Gunakan Half Kelly ({kelly['half_kelly']:.1%}) untuk optimal growth dengan lower volatility."
                    ])
                )
            elif risk_per_trade > kelly['half_kelly']:
                insights.append(
                    html.Li([
                        html.I(className="bi bi-info-circle text-info me-2"),
                        html.Strong("Between Half-Full Kelly: "),
                        f"Risk Anda ({risk_per_trade:.1%}) antara Half Kelly ({kelly['half_kelly']:.1%}) dan Full Kelly ({kelly['full_kelly']:.1%}). "
                        "Ini agresif tapi masih dalam range acceptable."
                    ])
                )
            else:
                insights.append(
                    html.Li([
                        html.I(className="bi bi-check-circle text-success me-2"),
                        html.Strong("Conservative Sizing: "),
                        f"Risk Anda ({risk_per_trade:.1%}) di bawah Half Kelly ({kelly['half_kelly']:.1%}). "
                        "Ini conservative dan aman, tapi bisa consider slight increase untuk faster growth."
                    ])
                )
        
        # Action items
        action_items = []
        
        if prob_ruin < 0.05 and percentile_95_dd < 20:
            action_items.append(
                html.Li([
                    html.Strong("✅ Implement: "),
                    f"Risk {risk_per_trade:.1%} per trade aman dan optimal. Implement dengan confidence!"
                ])
            )
        elif prob_ruin < 0.15:
            action_items.append(
                html.Li([
                    html.Strong("⚠️ Review: "),
                    "Risk level moderate. Review risk tolerance Anda dan pertimbangkan slight reduction."
                ])
            )
        else:
            action_items.append(
                html.Li([
                    html.Strong("❌ Reduce: "),
                    f"URGENT: Turunkan risk dari {risk_per_trade:.1%} ke {risk_per_trade * 0.5:.1%} atau kurang!"
                ])
            )
        
        action_items.append(
            html.Li([
                html.Strong("📊 Monitor: "),
                "Track actual performance vs projected. Re-run simulation setiap 3-6 bulan."
            ])
        )
        
        if not np.isnan(kelly['half_kelly']):
            action_items.append(
                html.Li([
                    html.Strong("🎯 Kelly Guidance: "),
                    f"Half Kelly ({kelly['half_kelly']:.1%}) adalah sweet spot untuk most traders. "
                    f"Quarter Kelly ({kelly['quarter_kelly']:.1%}) jika sangat risk-averse."
                ])
            )
        
        return html.Div([
            html.H6("📊 Insights:", className="mb-2"),
            html.Ul(insights, className="mb-3"),
            html.H6("💡 Action Items:", className="mb-2"),
            html.Ul(action_items, className="mb-0")
        ])
    
    
    logger.info("Monte Carlo callbacks registered successfully")


# Import html for button content
from dash import html
