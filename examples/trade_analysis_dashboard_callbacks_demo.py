"""
Trade Analysis Dashboard Callbacks Demo
Demonstrates the complete Trade Analysis Dashboard with all callbacks

This demo shows:
1. CSV file loading
2. Summary metrics display
3. Equity curve with drawdown shading
4. R-multiple distribution
5. MAE/MFE analysis
6. Time-based performance
7. Trade type analysis
8. Consecutive trades analysis
9. Risk metrics
10. Trade table with row selection
11. Chart click interactions
12. Export functionality
13. Navigation to next page

Requirements: 0.1, 0.10, 0.11
"""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output

from config.config import DASH_HOST, DASH_PORT

# Initialize app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "Trade Analysis Dashboard Demo"

# Create layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Trade Analysis Dashboard Demo", className="text-center my-4"),
            html.P(
                "Complete demonstration of Trade Analysis Dashboard callbacks",
                className="text-center text-muted mb-4"
            )
        ])
    ]),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Trade Analysis Dashboard", tab_id="trade-analysis"),
        dbc.Tab(label="Probability Explorer", tab_id="probability-explorer"),
    ], id="main-tabs", active_tab="trade-analysis"),
    
    # Content
    html.Div(id="tab-content", className="mt-4"),
    
    # Stores
    dcc.Store(id="global-data-store"),
    dcc.Store(id="feature-data-store"),
    dcc.Store(id="trade-data-store"),
    dcc.Store(id="merged-data-store"),
    
], fluid=True)


# Tab content callback
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'active_tab')]
)
def render_tab_content(active_tab):
    """Render content for active tab"""
    if active_tab == 'trade-analysis':
        from frontend.layouts.trade_analysis_dashboard_layout import create_trade_analysis_dashboard_layout
        return create_trade_analysis_dashboard_layout()
    elif active_tab == 'probability-explorer':
        return html.Div([
            html.H3("Probability Explorer"),
            html.P("This tab would show probability analysis features")
        ])
    
    return html.Div("Select a tab")


# Register callbacks
from frontend.callbacks.trade_analysis_dashboard_callbacks import register_trade_analysis_callbacks
register_trade_analysis_callbacks(app)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Trade Analysis Dashboard Callbacks Demo")
    print("="*60)
    print("\nFeatures demonstrated:")
    print("  1. ✓ CSV file upload and loading")
    print("  2. ✓ Summary metrics cards (6 metrics)")
    print("  3. ✓ Equity curve with drawdown shading")
    print("  4. ✓ R-multiple distribution with statistics")
    print("  5. ✓ MAE/MFE scatter plot analysis")
    print("  6. ✓ Time-based performance (hourly/daily/session)")
    print("  7. ✓ Trade type analysis (BUY vs SELL)")
    print("  8. ✓ Consecutive trades analysis (streaks)")
    print("  9. ✓ Comprehensive risk metrics table")
    print(" 10. ✓ Sortable/filterable trade table")
    print(" 11. ✓ Trade row click for details")
    print(" 12. ✓ Chart click to jump to trade")
    print(" 13. ✓ Export report functionality")
    print(" 14. ✓ Navigate to next page")
    print("\nInstructions:")
    print("  1. Open browser to http://localhost:8050")
    print("  2. Click 'Load Sample' or upload a trade CSV file")
    print("  3. Explore all visualizations and interactions")
    print("  4. Click on charts to select trades")
    print("  5. Use export buttons to download reports")
    print("  6. Click 'Next: Feature Analysis' to navigate")
    print("\n" + "="*60)
    print("Starting server...")
    print("="*60 + "\n")
    
    app.run_server(
        host=DASH_HOST,
        port=DASH_PORT,
        debug=True
    )
