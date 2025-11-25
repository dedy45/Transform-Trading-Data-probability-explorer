"""
Composite Score Visualization Component
UI components for composite score analysis and filtering
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_composite_score_analyzer():
    """
    Create complete composite score analyzer dashboard
    
    Components:
    - Weight adjustment sliders
    - Composite score gauge
    - Component breakdown radar chart
    - Score distribution histogram
    - Recommendation badges
    - Threshold filter with metrics
    - Backtest results table
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-star-fill me-2"),
                "Composite Score Filter"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Info Alert
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                html.Strong("Composite Score: "),
                "Combines multiple probability indicators into a single master score (0-100) for filtering trades. ",
                "Higher scores = Higher probability setups. Adjust component weights to match your strategy priorities."
            ], color="info", className="mb-3"),
            
            # Weight Adjustment Panel
            dbc.Card([
                dbc.CardHeader([
                    html.H6([
                        html.I(className="bi bi-sliders me-2"),
                        "Component Weights"
                    ], className="mb-0 d-inline"),
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-1"), "Reset"],
                        id="cs-reset-weights-btn",
                        size="sm",
                        color="secondary",
                        outline=True,
                        className="float-end"
                    )
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Win Rate (30%)", className="small fw-bold"),
                            dcc.Slider(
                                id='cs-weight-win-rate',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.30,
                                marks={0: '0%', 0.5: '50%', 1: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("Expected R (25%)", className="small fw-bold"),
                            dcc.Slider(
                                id='cs-weight-expected-r',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.25,
                                marks={0: '0%', 0.5: '50%', 1: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("Structure Quality (15%)", className="small fw-bold"),
                            dcc.Slider(
                                id='cs-weight-structure',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.15,
                                marks={0: '0%', 0.5: '50%', 1: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4)
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Time-Based (10%)", className="small fw-bold"),
                            dcc.Slider(
                                id='cs-weight-time',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.10,
                                marks={0: '0%', 0.5: '50%', 1: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("Correlation (10%)", className="small fw-bold"),
                            dcc.Slider(
                                id='cs-weight-correlation',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.10,
                                marks={0: '0%', 0.5: '50%', 1: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("Entry Quality (10%)", className="small fw-bold"),
                            dcc.Slider(
                                id='cs-weight-entry',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.10,
                                marks={0: '0%', 0.5: '50%', 1: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4)
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="cs-weight-sum-validation")
                        ], md=6),
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-calculator me-2"), "Hitung Ulang"],
                                id="cs-recalculate-btn",
                                color="primary",
                                className="float-end"
                            )
                        ], md=6)
                    ])
                ])
            ], className="mb-3"),
            
            # Visualization Row
            dbc.Row([
                # Gauge and Stats
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-cs-gauge",
                                type="default",
                                children=[
                                    dcc.Graph(
                                        id='cs-gauge-chart',
                                        config={'displayModeBar': False, 'displaylogo': False},
                                        style={'height': '250px'}
                                    )
                                ]
                            ),
                            html.Div(id="cs-gauge-stats", className="text-center")
                        ])
                    ])
                ], md=4),
                
                # Radar Chart
                dbc.Col([
                    html.H6("Component Breakdown", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-cs-radar",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='cs-radar-chart',
                                config={'displayModeBar': False, 'displaylogo': False},
                                style={'height': '350px'}
                            )
                        ]
                    )
                ], md=4),
                
                # Histogram
                dbc.Col([
                    html.H6("Score Distribution", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-cs-histogram",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='cs-histogram-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '300px'}
                            )
                        ]
                    ),
                    html.Div(id="cs-recommendation-badges", className="mt-2")
                ], md=4)
            ], className="mb-3"),
            
            # Threshold Filter Section
            dbc.Card([
                dbc.CardHeader([
                    html.H6([
                        html.I(className="bi bi-funnel me-2"),
                        "Score Threshold Filter"
                    ], className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Minimum Score Threshold", className="fw-bold mb-2"),
                            dcc.Slider(
                                id='cs-threshold-slider',
                                min=0,
                                max=100,
                                step=5,
                                value=60,
                                marks={
                                    0: '0',
                                    40: '40 (AVOID)',
                                    60: '60 (BUY)',
                                    80: '80 (STRONG)',
                                    100: '100'
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=8),
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-bar-chart me-2"), "Jalankan Backtest"],
                                id="cs-backtest-btn",
                                color="success",
                                size="lg",
                                className="w-100 mt-4"
                            )
                        ], md=4)
                    ], className="mb-3"),
                    
                    # Threshold Metrics
                    html.Div(id="cs-threshold-metrics", className="mb-3"),
                    
                    # Backtest Results Table
                    html.Div([
                        html.H6("Backtest Results", className="fw-bold mb-2"),
                        html.Div(id="cs-backtest-table")
                    ])
                ])
            ])
        ])
    ], className="mb-4")
