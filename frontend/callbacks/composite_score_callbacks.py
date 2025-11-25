"""
Composite Score Callbacks
Dash callbacks for composite score analysis and filtering
"""
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io

from backend.calculators.composite_score import (
    calculate_composite_score,
    backtest_score_threshold,
    filter_by_score,
    get_score_statistics,
    add_recommendation_labels,
    DEFAULT_WEIGHTS
)
from frontend.utils.error_handlers import (
    log_error
)


def register_composite_score_callbacks(app):
    """
    Register all callbacks for composite score analysis
    
    Parameters:
    -----------
    app : dash.Dash
        Dash application instance
    """
    
    @app.callback(
        Output('cs-weight-sum-validation', 'children'),
        [Input('cs-weight-win-rate', 'value'),
         Input('cs-weight-expected-r', 'value'),
         Input('cs-weight-structure', 'value'),
         Input('cs-weight-time', 'value'),
         Input('cs-weight-correlation', 'value'),
         Input('cs-weight-entry', 'value')]
    )
    def validate_weight_sum(w1, w2, w3, w4, w5, w6):
        """
        Validate that weights sum to 1.0
        
        Validates: Requirements 4.1
        Property 12: Component Weight Sum
        """
        total = w1 + w2 + w3 + w4 + w5 + w6
        
        if abs(total - 1.0) < 0.01:
            return dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                f"Total Bobot: {total:.2f} ✓"
            ], color="success", className="mb-0 py-2")
        else:
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Total Bobot: {total:.2f} (harus = 1.00)"
            ], color="warning", className="mb-0 py-2")
    
    
    @app.callback(
        [Output('cs-weight-win-rate', 'value'),
         Output('cs-weight-expected-r', 'value'),
         Output('cs-weight-structure', 'value'),
         Output('cs-weight-time', 'value'),
         Output('cs-weight-correlation', 'value'),
         Output('cs-weight-entry', 'value')],
        Input('cs-reset-weights-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def reset_weights(n_clicks):
        """Reset weights to default values"""
        if not n_clicks:
            raise PreventUpdate
        
        return (
            DEFAULT_WEIGHTS['win_rate'],
            DEFAULT_WEIGHTS['expected_r'],
            DEFAULT_WEIGHTS['structure_quality'],
            DEFAULT_WEIGHTS['time_based'],
            DEFAULT_WEIGHTS['correlation'],
            DEFAULT_WEIGHTS['entry_quality']
        )
    
    
    @app.callback(
        Output('composite-score-results-store', 'data'),
        [Input('merged-data-store', 'data'),
         Input('cs-recalculate-btn', 'n_clicks')],
        [State('cs-weight-win-rate', 'value'),
         State('cs-weight-expected-r', 'value'),
         State('cs-weight-structure', 'value'),
         State('cs-weight-time', 'value'),
         State('cs-weight-correlation', 'value'),
         State('cs-weight-entry', 'value')]
    )
    def calculate_scores(merged_data, n_clicks, w1, w2, w3, w4, w5, w6):
        """
        Calculate composite scores for all trades
        
        Triggered on data load or weight change.
        Validates: Requirements 4.1, 4.2
        Property 11: Composite Score Range Invariant
        Property 12: Component Weight Sum
        """
        if merged_data is None:
            raise PreventUpdate
        
        # Load data
        try:
            if isinstance(merged_data, dict) and 'data' in merged_data:
                df = pd.read_json(io.StringIO(merged_data['data']), orient='split')
            else:
                df = pd.DataFrame(merged_data)
        except Exception as e:
            log_error(e, "Composite score data loading")
            raise PreventUpdate
        
        if df.empty:
            raise PreventUpdate
        
        # Validate weights sum to 1.0
        weights = {
            'win_rate': w1,
            'expected_r': w2,
            'structure_quality': w3,
            'time_based': w4,
            'correlation': w5,
            'entry_quality': w6
        }
        
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # Return error state
            return {
                'error': f'Weights must sum to 1.0, got {weight_sum:.2f}',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Calculate composite scores
            scored_df = calculate_composite_score(df, weights=weights)
            
            # Add recommendation labels
            scored_df = add_recommendation_labels(scored_df)
            
            # Get statistics
            stats = get_score_statistics(scored_df)
            
            # Store results
            results = {
                'scores_data': scored_df.to_json(orient='split'),
                'statistics': stats,
                'weights': weights,
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            log_error(e, "Composite score calculation", include_traceback=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    
    @app.callback(
        [Output('cs-gauge-chart', 'figure'),
         Output('cs-gauge-stats', 'children')],
        Input('composite-score-results-store', 'data')
    )
    def update_gauge(results):
        """
        Update composite score gauge chart
        
        Validates: Requirements 4.3
        """
        if results is None or 'error' in results:
            # Return empty gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=0,
                title={'text': "Skor Komposit"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "lightgray"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(255,0,0,0.2)"},
                        {'range': [40, 60], 'color': "rgba(255,255,0,0.2)"},
                        {'range': [60, 80], 'color': "rgba(0,255,0,0.2)"},
                        {'range': [80, 100], 'color': "rgba(0,128,0,0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            
            error_msg = results.get('error', 'No data available') if results else 'No data available'
            stats = html.Div([
                html.Small(error_msg, className="text-muted")
            ])
            
            return fig, stats
        
        # Get statistics
        stats_data = results.get('statistics', {})
        mean_score = stats_data.get('mean_score', 0)
        
        # Determine color based on score
        if mean_score >= 80:
            color = "darkgreen"
        elif mean_score >= 60:
            color = "green"
        elif mean_score >= 40:
            color = "orange"
        else:
            color = "red"
        
        # Create gauge figure
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=mean_score,
            title={'text': "Skor Komposit Rata-rata"},
            delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(255,0,0,0.2)"},
                    {'range': [40, 60], 'color': "rgba(255,255,0,0.2)"},
                    {'range': [60, 80], 'color': "rgba(0,255,0,0.2)"},
                    {'range': [80, 100], 'color': "rgba(0,128,0,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': mean_score
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        
        # Create stats display
        median_score = stats_data.get('median_score', 0)
        std_score = stats_data.get('std_score', 0)
        
        stats = html.Div([
            html.Small([
                html.Strong("Median: "),
                f"{median_score:.1f} | ",
                html.Strong("Std Dev: "),
                f"{std_score:.1f}"
            ], className="text-muted")
        ])
        
        return fig, stats
    
    
    @app.callback(
        Output('cs-radar-chart', 'figure'),
        Input('composite-score-results-store', 'data')
    )
    def update_radar(results):
        """
        Update component breakdown radar chart
        
        Validates: Requirements 4.4
        """
        if results is None or 'error' in results:
            # Return empty radar
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[0, 0, 0, 0, 0, 0],
                theta=['Win Rate', 'Expected R', 'Structure', 'Time', 'Correlation', 'Entry'],
                fill='toself',
                name='Component Scores'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=350,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return fig
        
        # Load scored data
        try:
            scores_json = results.get('scores_data')
            if not scores_json:
                raise ValueError("No scores data")
            
            df = pd.read_json(io.StringIO(scores_json), orient='split')
            
            # Calculate average component scores
            components = {
                'Win Rate': df.get('score_win_rate', pd.Series([50])).mean(),
                'Expected R': df.get('score_expected_r', pd.Series([50])).mean(),
                'Structure': df.get('score_structure_quality', pd.Series([50])).mean(),
                'Time': df.get('score_time_based', pd.Series([50])).mean(),
                'Correlation': df.get('score_correlation', pd.Series([50])).mean(),
                'Entry': df.get('score_entry_quality', pd.Series([50])).mean()
            }
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(components.values()),
                theta=list(components.keys()),
                fill='toself',
                name='Avg Component Scores',
                line=dict(color='rgb(0, 123, 255)', width=2),
                fillcolor='rgba(0, 123, 255, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickmode='linear',
                        tick0=0,
                        dtick=20
                    )
                ),
                showlegend=False,
                height=350,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
            
        except Exception as e:
            log_error(e, "Radar chart creation")
            # Return empty radar
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[0, 0, 0, 0, 0, 0],
                theta=['Win Rate', 'Expected R', 'Structure', 'Time', 'Correlation', 'Entry'],
                fill='toself'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=350,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return fig
    
    
    @app.callback(
        Output('cs-histogram-chart', 'figure'),
        Input('composite-score-results-store', 'data')
    )
    def update_histogram(results):
        """
        Update score distribution histogram
        
        Validates: Requirements 4.3
        """
        if results is None or 'error' in results:
            # Return empty histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=[], nbinsx=20))
            fig.update_layout(
                title="Score Distribution",
                xaxis_title="Composite Score",
                yaxis_title="Count",
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return fig
        
        # Load scored data
        try:
            scores_json = results.get('scores_data')
            if not scores_json:
                raise ValueError("No scores data")
            
            df = pd.read_json(io.StringIO(scores_json), orient='split')
            
            if 'composite_score' not in df.columns:
                raise ValueError("No composite_score column")
            
            scores = df['composite_score'].dropna()
            
            # Create histogram with color coding
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=20,
                marker=dict(
                    color=scores,
                    colorscale=[
                        [0, 'red'],
                        [0.4, 'orange'],
                        [0.6, 'yellow'],
                        [0.8, 'lightgreen'],
                        [1, 'darkgreen']
                    ],
                    cmin=0,
                    cmax=100,
                    colorbar=dict(title="Score")
                ),
                hovertemplate='Score: %{x:.1f}<br>Count: %{y}<extra></extra>'
            ))
            
            # Add threshold lines
            fig.add_vline(x=40, line_dash="dash", line_color="red", opacity=0.5,
                         annotation_text="AVOID", annotation_position="top")
            fig.add_vline(x=60, line_dash="dash", line_color="orange", opacity=0.5,
                         annotation_text="NEUTRAL", annotation_position="top")
            fig.add_vline(x=80, line_dash="dash", line_color="green", opacity=0.5,
                         annotation_text="STRONG BUY", annotation_position="top")
            
            fig.update_layout(
                title="Distribusi Skor Komposit",
                xaxis_title="Composite Score",
                yaxis_title="Jumlah Trading",
                height=300,
                margin=dict(l=40, r=40, t=40, b=40),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            log_error(e, "Histogram creation")
            # Return empty histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=[], nbinsx=20))
            fig.update_layout(
                title="Score Distribution",
                xaxis_title="Composite Score",
                yaxis_title="Count",
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return fig
    
    
    @app.callback(
        Output('cs-recommendation-badges', 'children'),
        Input('composite-score-results-store', 'data')
    )
    def update_recommendations(results):
        """
        Update recommendation labels display
        
        Validates: Requirements 4.4
        """
        if results is None or 'error' in results:
            # Return empty badges
            return dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Badge("0", color="success", className="fs-4"),
                        html.Div("STRONG BUY", className="small text-muted mt-1"),
                        html.Div("(≥80)", className="small text-muted")
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    html.Div([
                        dbc.Badge("0", color="info", className="fs-4"),
                        html.Div("BUY", className="small text-muted mt-1"),
                        html.Div("(60-79)", className="small text-muted")
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    html.Div([
                        dbc.Badge("0", color="warning", className="fs-4"),
                        html.Div("NEUTRAL", className="small text-muted mt-1"),
                        html.Div("(40-59)", className="small text-muted")
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    html.Div([
                        dbc.Badge("0", color="danger", className="fs-4"),
                        html.Div("AVOID", className="small text-muted mt-1"),
                        html.Div("(<40)", className="small text-muted")
                    ], className="text-center")
                ], width=3)
            ])
        
        # Get statistics
        stats = results.get('statistics', {})
        
        # Get counts and percentages
        pct_strong_buy = stats.get('pct_strong_buy', 0)
        pct_buy = stats.get('pct_buy', 0)
        pct_neutral = stats.get('pct_neutral', 0)
        pct_avoid = stats.get('pct_avoid', 0)
        
        # Load data to get actual counts
        try:
            scores_json = results.get('scores_data')
            if scores_json:
                df = pd.read_json(io.StringIO(scores_json), orient='split')
                total = len(df)
                
                count_strong_buy = int((pct_strong_buy / 100) * total)
                count_buy = int((pct_buy / 100) * total)
                count_neutral = int((pct_neutral / 100) * total)
                count_avoid = int((pct_avoid / 100) * total)
            else:
                count_strong_buy = count_buy = count_neutral = count_avoid = 0
        except:
            count_strong_buy = count_buy = count_neutral = count_avoid = 0
        
        return dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Badge(f"{count_strong_buy}", color="success", className="fs-4"),
                    html.Div("STRONG BUY", className="small text-muted mt-1"),
                    html.Div(f"(≥80) {pct_strong_buy:.1f}%", className="small text-muted")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    dbc.Badge(f"{count_buy}", color="info", className="fs-4"),
                    html.Div("BUY", className="small text-muted mt-1"),
                    html.Div(f"(60-79) {pct_buy:.1f}%", className="small text-muted")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    dbc.Badge(f"{count_neutral}", color="warning", className="fs-4"),
                    html.Div("NEUTRAL", className="small text-muted mt-1"),
                    html.Div(f"(40-59) {pct_neutral:.1f}%", className="small text-muted")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    dbc.Badge(f"{count_avoid}", color="danger", className="fs-4"),
                    html.Div("AVOID", className="small text-muted mt-1"),
                    html.Div(f"(<40) {pct_avoid:.1f}%", className="small text-muted")
                ], className="text-center")
            ], width=3)
        ])
    
    
    @app.callback(
        Output('cs-threshold-metrics', 'children'),
        Input('cs-threshold-slider', 'value'),
        State('composite-score-results-store', 'data')
    )
    def update_threshold_metrics(threshold, results):
        """
        Update metrics for filtered trades based on threshold
        
        Uses debouncing to avoid excessive calculations.
        Validates: Requirements 4.5
        Property 13: Score Threshold Filtering
        """
        if results is None or 'error' in results:
            return dbc.Alert(
                "Hitung skor komposit terlebih dahulu",
                color="warning",
                className="mb-0"
            )
        
        # Load scored data
        try:
            scores_json = results.get('scores_data')
            if not scores_json:
                raise ValueError("No scores data")
            
            df = pd.read_json(io.StringIO(scores_json), orient='split')
            
            # Filter by threshold
            filtered_df = filter_by_score(df, threshold)
            
            # Calculate metrics
            n_total = len(df)
            n_filtered = len(filtered_df)
            pct_filtered = (n_filtered / n_total * 100) if n_total > 0 else 0
            
            if n_filtered == 0:
                return dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    f"Tidak ada trading dengan skor ≥ {threshold}"
                ], color="warning", className="mb-0")
            
            # Calculate win rate
            if 'R_multiple' in filtered_df.columns:
                winners = filtered_df[filtered_df['R_multiple'] > 0]
                win_rate = (len(winners) / n_filtered * 100) if n_filtered > 0 else 0
                expectancy = filtered_df['R_multiple'].mean()
                avg_r = filtered_df['R_multiple'].mean()
            else:
                win_rate = 0
                expectancy = 0
                avg_r = 0
            
            # Determine if this is a good threshold
            is_good = win_rate > 55 and expectancy > 0.5
            
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4(f"{n_filtered:,}", className="mb-0"),
                            html.Small("Trading", className="text-muted")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H4(f"{pct_filtered:.1f}%", className="mb-0"),
                            html.Small("dari Total", className="text-muted")
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
                            html.H4(f"{expectancy:.2f}R", className="mb-0"),
                            html.Small("Expectancy", className="text-muted")
                        ], className="text-center")
                    ], width=3)
                ], className="mb-3"),
                
                dbc.Alert([
                    html.I(className=f"bi bi-{'check-circle' if is_good else 'info-circle'} me-2"),
                    html.Strong("Rekomendasi: "),
                    f"Threshold {threshold} {'sangat baik' if is_good else 'cukup baik' if win_rate > 50 else 'perlu ditingkatkan'} "
                    f"dengan win rate {win_rate:.1f}% dan expectancy {expectancy:.2f}R"
                ], color="success" if is_good else "info" if win_rate > 50 else "warning", className="mb-0")
            ])
            
        except Exception as e:
            log_error(e, "Threshold metrics calculation")
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Error: "),
                str(e)
            ], color="danger", className="mb-0")
    
    
    @app.callback(
        Output('cs-backtest-table', 'children'),
        Input('cs-backtest-btn', 'n_clicks'),
        State('composite-score-results-store', 'data'),
        prevent_initial_call=True
    )
    def run_backtest(n_clicks, results):
        """
        Run backtest on multiple thresholds
        
        Tests thresholds [50, 60, 70, 80] and displays results.
        Highlights optimal threshold (max expectancy).
        Validates: Requirements 4.6, 10.4
        Property 14: Backtest Threshold Monotonicity
        Property 26: Optimal Threshold Identification
        """
        if not n_clicks or results is None or 'error' in results:
            return dbc.Alert(
                "Hitung skor komposit terlebih dahulu",
                color="warning",
                className="mb-0"
            )
        
        # Load scored data
        try:
            scores_json = results.get('scores_data')
            if not scores_json:
                raise ValueError("No scores data")
            
            df = pd.read_json(io.StringIO(scores_json), orient='split')
            
            # Run backtest
            backtest_results = backtest_score_threshold(df, thresholds=[50, 60, 70, 80])
            
            if backtest_results.empty:
                return dbc.Alert(
                    "Tidak dapat menjalankan backtest. Pastikan data memiliki kolom R_multiple.",
                    color="warning",
                    className="mb-0"
                )
            
            # Find optimal threshold (max expectancy)
            valid_results = backtest_results.dropna(subset=['expectancy'])
            if len(valid_results) > 0:
                optimal_idx = valid_results['expectancy'].idxmax()
                optimal_threshold = valid_results.loc[optimal_idx, 'threshold']
            else:
                optimal_threshold = None
            
            # Create table
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Threshold"),
                    html.Th("Win Rate"),
                    html.Th("Expectancy"),
                    html.Th("Trade Count"),
                    html.Th("Trade %"),
                    html.Th("Avg Score")
                ]))
            ]
            
            table_rows = []
            for idx, row in backtest_results.iterrows():
                is_optimal = (row['threshold'] == optimal_threshold)
                
                # Format values
                threshold = f"{row['threshold']:.0f}"
                win_rate = f"{row['win_rate']*100:.1f}%" if not pd.isna(row['win_rate']) else "N/A"
                expectancy = f"{row['expectancy']:.2f}R" if not pd.isna(row['expectancy']) else "N/A"
                trade_count = f"{row['trade_frequency']:.0f}"
                trade_pct = f"{row['trade_frequency_pct']:.1f}%"
                avg_score = f"{row['avg_score']:.1f}" if not pd.isna(row['avg_score']) else "N/A"
                
                # Highlight optimal row
                row_class = "table-success fw-bold" if is_optimal else ""
                
                table_rows.append(html.Tr([
                    html.Td([threshold, " ⭐" if is_optimal else ""]),
                    html.Td(win_rate),
                    html.Td(expectancy),
                    html.Td(trade_count),
                    html.Td(trade_pct),
                    html.Td(avg_score)
                ], className=row_class))
            
            table_body = [html.Tbody(table_rows)]
            
            table = dbc.Table(
                table_header + table_body,
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                size="sm"
            )
            
            # Add summary
            if optimal_threshold is not None:
                optimal_row = backtest_results[backtest_results['threshold'] == optimal_threshold].iloc[0]
                summary = dbc.Alert([
                    html.I(className="bi bi-trophy me-2"),
                    html.Strong("Threshold Optimal: "),
                    f"{optimal_threshold:.0f} dengan expectancy {optimal_row['expectancy']:.2f}R "
                    f"dan win rate {optimal_row['win_rate']*100:.1f}% "
                    f"({optimal_row['trade_frequency']:.0f} trading)"
                ], color="success", className="mt-3 mb-0")
            else:
                summary = dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    "Tidak ada threshold dengan data yang cukup"
                ], color="info", className="mt-3 mb-0")
            
            return html.Div([table, summary])
            
        except Exception as e:
            log_error(e, "Backtest execution", include_traceback=True)
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Error menjalankan backtest: "),
                str(e)
            ], color="danger", className="mb-0")
