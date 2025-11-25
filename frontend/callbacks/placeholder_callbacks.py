"""
Placeholder Callbacks

Provides default/empty callbacks for all components that don't have data yet.
This prevents the "callback not found" errors that cause blank pages.
"""

from dash import Input, Output, html, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


def register_placeholder_callbacks(app):
    """Register placeholder callbacks for all components."""
    
    # Empty figure for charts
    empty_fig = go.Figure()
    empty_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[{
            'text': 'Upload data to see visualization',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16, 'color': 'gray'}
        }]
    )
    
    # Placeholder for all chart outputs
    # MODIFIED - These are now disabled to avoid conflicts with trade_analysis_dashboard_callbacks.py
    # The trade analysis dashboard handles all chart updates when data is available
    chart_ids = []
    
    for chart_id in chart_ids:
        try:
            @app.callback(
                Output(chart_id, 'figure'),
                Input('trade-data-loaded-store', 'data'),
                prevent_initial_call=False
            )
            def update_chart(data):
                if not data:
                    return empty_fig
                return empty_fig
        except:
            pass
    
    # Placeholder for text outputs
    # MODIFIED - These are now disabled to avoid conflicts with trade_analysis_dashboard_callbacks.py
    # The trade analysis dashboard handles all summary text updates when data is available
    text_ids = [
        # 'summary-total-trades',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'summary-win-rate',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'summary-avg-r',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'summary-expectancy',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'summary-max-dd',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'summary-profit-factor'  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
    ]
    
    for text_id in text_ids:
        try:
            @app.callback(
                Output(text_id, 'children'),
                Input('trade-data-loaded-store', 'data'),
                prevent_initial_call=False
            )
            def update_text(data):
                return "0"
        except:
            pass
    
    # Placeholder for div outputs
    # MODIFIED - These are now disabled to avoid conflicts with trade_analysis_dashboard_callbacks.py
    # The trade analysis dashboard handles all div updates when data is available
    div_ids = [
        # 'r-statistics-table',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'winners-stats-table',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'losers-stats-table',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'time-based-content',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'risk-metrics-table',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'trade-table-container',  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
        # 'upload-status'  # REMOVED - Handled by trade_analysis_dashboard_callbacks.py
    ]
    
    for div_id in div_ids:
        try:
            @app.callback(
                Output(div_id, 'children'),
                Input('trade-data-loaded-store', 'data'),
                prevent_initial_call=False
            )
            def update_div(data):
                return html.Div("No data loaded", className="text-muted")
        except:
            pass
    
    print("[OK] Placeholder callbacks registered")


print("[OK] Placeholder callbacks module loaded")
