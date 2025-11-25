"""
MAE/MFE Optimizer Component
UI components for MAE/MFE analysis and SL/TP optimization
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_mae_mfe_optimizer():
    """
    Create complete MAE/MFE optimizer dashboard
    
    Components:
    - MAE/MFE scatter plot
    - SL optimizer with slider
    - TP optimizer with slider
    - Pattern detection alerts
    - Profit left analysis
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-bullseye me-2"),
                "MAE/MFE Optimizer"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Info Alert
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                html.Strong("MAE/MFE Analysis: "),
                "Optimize your stop loss and take profit levels based on how your trades actually moved. ",
                "MAE = Maximum Adverse Excursion (max drawdown), MFE = Maximum Favorable Excursion (max profit)."
            ], color="info", className="mb-3"),
            
            # Scatter Plot and Profit Left
            dbc.Row([
                dbc.Col([
                    html.H6("MAE vs MFE Scatter Plot", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-mae-mfe-scatter",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='mae-mfe-scatter-plot',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '400px'}
                            )
                        ]
                    )
                ], md=8),
                dbc.Col([
                    html.H6("Profit Left Analysis", className="fw-bold mb-2"),
                    html.Div(id="mae-mfe-profit-left-display")
                ], md=4)
            ], className="mb-4"),
            
            # Pattern Alerts
            dbc.Row([
                dbc.Col([
                    html.H6("Pattern Detection", className="fw-bold mb-2"),
                    html.Div(id="mae-mfe-pattern-alerts")
                ])
            ], className="mb-4"),
            
            # SL and TP Optimizers
            dbc.Row([
                # SL Optimizer
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="bi bi-shield-x me-2"),
                                "Stop Loss Optimizer"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Label("SL Level (R-multiples)", className="fw-bold mb-2"),
                            dcc.Slider(
                                id='mae-mfe-sl-slider',
                                min=0.3,
                                max=2.0,
                                step=0.1,
                                value=1.0,
                                marks={
                                    0.3: '0.3R',
                                    0.5: '0.5R',
                                    1.0: '1.0R',
                                    1.5: '1.5R',
                                    2.0: '2.0R'
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Div(id="mae-mfe-sl-results", className="mt-3"),
                            html.Div(id="mae-mfe-sl-recommendation", className="mt-2")
                        ])
                    ])
                ], md=6),
                
                # TP Optimizer
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="bi bi-trophy me-2"),
                                "Take Profit Optimizer"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Label("TP Level (R-multiples)", className="fw-bold mb-2"),
                            dcc.Slider(
                                id='mae-mfe-tp-slider',
                                min=0.5,
                                max=5.0,
                                step=0.5,
                                value=2.0,
                                marks={
                                    0.5: '0.5R',
                                    1.0: '1.0R',
                                    2.0: '2.0R',
                                    3.0: '3.0R',
                                    5.0: '5.0R'
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Div(id="mae-mfe-tp-results", className="mt-3"),
                            html.Div(id="mae-mfe-tp-recommendation", className="mt-2")
                        ])
                    ])
                ], md=6)
            ])
        ])
    ], className="mb-4")
