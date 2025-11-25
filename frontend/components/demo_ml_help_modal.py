"""
Demo script for ML Help Modal Component

Run this script to test the help modal in isolation:
    python frontend/components/demo_ml_help_modal.py
"""

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from ml_help_modal import get_help_content

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

# Create modal layout
help_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle([
        html.I(className="bi bi-question-circle me-2"),
        "ML Prediction Engine - Help & Documentation"
    ])),
    dbc.ModalBody([
        dbc.Tabs([
            dbc.Tab(
                label="Overview",
                tab_id="help-overview",
                label_style={"cursor": "pointer"},
                active_label_style={"fontWeight": "bold"}
            ),
            dbc.Tab(
                label="How to Use",
                tab_id="help-howto",
                label_style={"cursor": "pointer"},
                active_label_style={"fontWeight": "bold"}
            ),
            dbc.Tab(
                label="Interpretation Guide",
                tab_id="help-interpretation",
                label_style={"cursor": "pointer"},
                active_label_style={"fontWeight": "bold"}
            ),
            dbc.Tab(
                label="FAQ",
                tab_id="help-faq",
                label_style={"cursor": "pointer"},
                active_label_style={"fontWeight": "bold"}
            ),
        ], id='ml-help-tabs', active_tab='help-overview'),
        
        html.Hr(),
        
        html.Div(id='ml-help-content', style={'maxHeight': '60vh', 'overflowY': 'auto'})
    ]),
    dbc.ModalFooter([
        dbc.Button("Close", id="ml-help-close-btn", className="ms-auto", color="secondary")
    ])
], id='ml-help-modal', size='xl', is_open=True, scrollable=True)

# App layout
app.layout = dbc.Container([
    html.H1("ML Help Modal Demo", className="my-4"),
    
    dbc.Button([
        html.I(className="bi bi-question-circle me-2"),
        "Open Help"
    ], id='ml-help-open-btn', color='info', className="mb-4"),
    
    help_modal,
    
    html.Div([
        html.H4("Instructions:", className="mt-4"),
        html.Ul([
            html.Li("Click 'Open Help' button to open the modal"),
            html.Li("Navigate through the 4 tabs: Overview, How to Use, Interpretation Guide, FAQ"),
            html.Li("Test scrolling within the modal body"),
            html.Li("Click 'Close' or outside the modal to close it"),
        ])
    ])
], fluid=True, className="py-4")


# Callbacks
@app.callback(
    Output('ml-help-modal', 'is_open'),
    [Input('ml-help-open-btn', 'n_clicks'),
     Input('ml-help-close-btn', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_help_modal(open_clicks, close_clicks):
    """Toggle help modal open/close"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'ml-help-open-btn':
        return True
    elif button_id == 'ml-help-close-btn':
        return False
    
    return False


@app.callback(
    Output('ml-help-content', 'children'),
    Input('ml-help-tabs', 'active_tab')
)
def update_help_content(active_tab):
    """Update help content based on active tab"""
    if not active_tab:
        return html.Div("Loading...", className="text-muted")
    
    return get_help_content(active_tab)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ML Help Modal Demo")
    print("="*60)
    print("\nStarting Dash server...")
    print("Open your browser and navigate to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run_server(debug=True, port=8050)
