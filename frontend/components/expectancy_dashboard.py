"""
Expectancy Dashboard Component
UI components for expectancy analysis visualization
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_expectancy_dashboard():
    """
    Create complete expectancy analysis dashboard
    
    Components:
    - Summary card with key metrics
    - Expectancy heatmap by market conditions
    - R-multiple distribution histogram
    - Expectancy evolution line chart
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-graph-up me-2"),
                "Analisis Expectancy"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Summary Card
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Expectancy (R)", className="text-muted mb-2"),
                                    html.H3(id="expectancy-summary-expectancy-r", children="--", 
                                           className="mb-1 text-primary"),
                                    html.Div(id="expectancy-summary-expectancy-status", 
                                            className="small")
                                ], width=3),
                                dbc.Col([
                                    html.H6("Win Rate", className="text-muted mb-2"),
                                    html.H3(id="expectancy-summary-win-rate", children="--", 
                                           className="mb-0")
                                ], width=2),
                                dbc.Col([
                                    html.H6("Avg Win (R)", className="text-muted mb-2"),
                                    html.H3(id="expectancy-summary-avg-win", children="--", 
                                           className="mb-0 text-success")
                                ], width=2),
                                dbc.Col([
                                    html.H6("Avg Loss (R)", className="text-muted mb-2"),
                                    html.H3(id="expectancy-summary-avg-loss", children="--", 
                                           className="mb-0 text-danger")
                                ], width=2),
                                dbc.Col([
                                    html.Div([
                                        html.I(className="bi bi-info-circle me-2"),
                                        html.Strong("Expectancy"),
                                        html.P("Expected profit per trade in R-multiples. "
                                              "Positive = profitable strategy.", 
                                              className="small text-muted mb-0 mt-1")
                                    ])
                                ], width=3)
                            ])
                        ])
                    ], className="bg-light")
                ])
            ], className="mb-3"),
            
            # Charts Row
            dbc.Row([
                # Expectancy Heatmap
                dbc.Col([
                    html.H6("Expectancy by Market Condition", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-expectancy-heatmap",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='expectancy-heatmap-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '350px'}
                            )
                        ]
                    )
                ], md=6),
                
                # R-Multiple Distribution
                dbc.Col([
                    html.H6("R-Multiple Distribution", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-expectancy-histogram",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='expectancy-histogram-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '300px'}
                            )
                        ]
                    ),
                    html.Div(id="expectancy-histogram-stats", className="mt-2")
                ], md=6)
            ], className="mb-3"),
            
            # Evolution Chart
            dbc.Row([
                dbc.Col([
                    html.H6("Expectancy Evolution Over Time", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-expectancy-evolution",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='expectancy-evolution-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '300px'}
                            )
                        ]
                    )
                ])
            ])
        ])
    ], className="mb-4")
