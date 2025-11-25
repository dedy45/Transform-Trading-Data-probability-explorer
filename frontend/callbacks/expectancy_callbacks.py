"""
Expectancy Callbacks

Callbacks for expectancy analysis:
- Calculate expectancy metrics on data load
- Update expectancy visualizations
- Handle expectancy-related user interactions
"""

import pandas as pd
import numpy as np
from dash import Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html

from backend.calculators.expectancy_calculator import (
    compute_expectancy_R,
    compute_expectancy_by_group,
    compute_r_percentiles
)
from frontend.utils.error_handlers import (
    validate_required_columns,
    log_error
)


def register_expectancy_callbacks(app):
    """
    Register all expectancy analysis callbacks.
    
    Args:
        app: Dash application instance
    
    NOTE: Main expectancy calculation and summary card updates are now integrated 
    into trade_analysis_dashboard_callbacks.py to avoid callback conflicts.
    This file only contains visualization callbacks (heatmap, histogram, evolution).
    """
    
    # Callback removed: calculate_expectancy_metrics
    # Now handled by update_all_visualizations in trade_analysis_dashboard_callbacks
    
    # Callback removed: update_expectancy_summary_card
    # Now handled by update_all_visualizations in trade_analysis_dashboard_callbacks
    
    @app.callback(
        Output('expectancy-heatmap-chart', 'figure'),
        Input('expectancy-results-store', 'data')
    )
    def update_expectancy_heatmap(results_store):
        """
        Update expectancy heatmap from grouped expectancy data.
        
        Requirements: 1.2, 1.4
        """
        import plotly.graph_objects as go
        
        if not results_store or 'grouped_expectancy' not in results_store:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )
            return fig
        
        if 'error' in results_store and results_store['error']:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {results_store['error']}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )
            return fig
        
        grouped_json = results_store.get('grouped_expectancy')
        if not grouped_json:
            fig = go.Figure()
            fig.add_annotation(
                text="No grouped data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )
            return fig
        
        try:
            import io
            grouped_df = pd.read_json(io.StringIO(grouped_json), orient='split')
            
            # Create bar chart (heatmap alternative for 1D data)
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=grouped_df['group_value'],
                y=grouped_df['expectancy_R'],
                marker=dict(
                    color=grouped_df['expectancy_R'],
                    colorscale='RdYlGn',
                    colorbar=dict(title="Expectancy (R)"),
                    line=dict(color='black', width=1)
                ),
                text=[f"{val:.3f}R" for val in grouped_df['expectancy_R']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Expectancy: %{y:.3f}R<br>Sample: %{customdata}<extra></extra>',
                customdata=grouped_df['sample_size']
            ))
            
            fig.update_layout(
                title="Expectancy by Market Condition",
                xaxis_title="Market Condition",
                yaxis_title="Expectancy (R)",
                hovermode='closest',
                plot_bgcolor='white',
                showlegend=False
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
            
        except Exception as e:
            log_error(e, "Expectancy heatmap creation")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )
            return fig
    
    @app.callback(
        [
            Output('expectancy-histogram-chart', 'figure'),
            Output('expectancy-histogram-stats', 'children')
        ],
        Input('expectancy-results-store', 'data')
    )
    def update_r_distribution_histogram(results_store):
        """
        Update R-multiple distribution histogram from percentiles.
        
        Requirements: 1.2, 1.4
        """
        import plotly.graph_objects as go
        
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        empty_fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        
        if not results_store or 'percentiles' not in results_store:
            return empty_fig, ""
        
        if 'error' in results_store and results_store['error']:
            return empty_fig, dbc.Alert(results_store['error'], color="danger", className="mt-2")
        
        percentiles = results_store.get('percentiles')
        if not percentiles:
            return empty_fig, ""
        
        try:
            # Create stats display
            stats_content = dbc.Row([
                dbc.Col([
                    html.Small("Mean:", className="text-muted"),
                    html.Div(f"{percentiles.get('mean_R', 0):.3f}R", className="fw-bold")
                ], width=3),
                dbc.Col([
                    html.Small("Median:", className="text-muted"),
                    html.Div(f"{percentiles.get('p50', 0):.3f}R", className="fw-bold")
                ], width=3),
                dbc.Col([
                    html.Small("Std Dev:", className="text-muted"),
                    html.Div(f"{percentiles.get('std_R', 0):.3f}R", className="fw-bold")
                ], width=3),
                dbc.Col([
                    html.Small("Range:", className="text-muted"),
                    html.Div(f"{percentiles.get('min_R', 0):.2f} to {percentiles.get('max_R', 0):.2f}R", 
                            className="fw-bold small")
                ], width=3)
            ])
            
            # Create simple percentile visualization
            fig = go.Figure()
            
            # Add percentile bars
            percentile_values = [
                ('P25', percentiles.get('p25', 0)),
                ('P50', percentiles.get('p50', 0)),
                ('P75', percentiles.get('p75', 0)),
                ('P90', percentiles.get('p90', 0)),
                ('P95', percentiles.get('p95', 0))
            ]
            
            fig.add_trace(go.Bar(
                x=[p[0] for p in percentile_values],
                y=[p[1] for p in percentile_values],
                marker=dict(
                    color=[p[1] for p in percentile_values],
                    colorscale='RdYlGn',
                    line=dict(color='black', width=1)
                ),
                text=[f"{p[1]:.3f}R" for p in percentile_values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="R-Multiple Percentiles",
                xaxis_title="Percentile",
                yaxis_title="R-Multiple",
                plot_bgcolor='white',
                showlegend=False
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig, stats_content
            
        except Exception as e:
            log_error(e, "R-distribution histogram creation")
            return empty_fig, dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Error: "),
                str(e)
            ], color="danger", className="mt-2")
    
    @app.callback(
        Output('expectancy-evolution-chart', 'figure'),
        [
            Input('merged-data-store', 'data'),
            Input('trade-data-loaded-store', 'data')
        ]
    )
    def update_expectancy_evolution(merged_data_store, trade_data_store):
        """
        Update expectancy evolution line chart over time.
        
        Requirements: 1.2, 1.4
        """
        import plotly.graph_objects as go
        
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        empty_fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        
        # Try merged-data-store first, then trade-data-loaded-store
        data_store = merged_data_store if merged_data_store and 'data' in merged_data_store else trade_data_store
        
        if not data_store or 'data' not in data_store:
            return empty_fig
        
        try:
            import io
            df = pd.read_json(io.StringIO(data_store['data']), orient='split')
            
            if df.empty or 'R_multiple' not in df.columns:
                return empty_fig
            
            # Calculate cumulative expectancy
            df = df.sort_index()
            df['cumulative_expectancy'] = df['R_multiple'].expanding().mean()
            
            # Create line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['cumulative_expectancy'],
                mode='lines',
                name='Cumulative Expectancy',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 255, 0.1)'
            ))
            
            fig.update_layout(
                title="Expectancy Evolution Over Time",
                xaxis_title="Trade Number",
                yaxis_title="Cumulative Expectancy (R)",
                plot_bgcolor='white',
                hovermode='x unified'
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
            
        except Exception as e:
            log_error(e, "Expectancy evolution chart creation")
            return empty_fig
    
    print("[OK] Expectancy callbacks registered")


print("[OK] Expectancy callbacks module loaded")
