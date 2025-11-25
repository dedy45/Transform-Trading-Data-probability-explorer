"""
Monte Carlo Visualization Component
UI components for Monte Carlo simulation visualization
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_monte_carlo_summary_cards():
    """Create summary cards for Monte Carlo results"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Median Final Equity", className="text-muted mb-2"),
                    html.H3(id="mc-median-equity-value", children="-", className="mb-0 text-primary")
                ])
            ], className="text-center")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("5th Percentile", className="text-muted mb-2"),
                    html.H3(id="mc-p5-equity-value", children="-", className="mb-0 text-danger")
                ])
            ], className="text-center")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("95th Percentile", className="text-muted mb-2"),
                    html.H3(id="mc-p95-equity-value", children="-", className="mb-0 text-success")
                ])
            ], className="text-center")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Prob Reach 2x", className="text-muted mb-2"),
                    html.H3(id="mc-prob-target-value", children="-", className="mb-0 text-info")
                ])
            ], className="text-center")
        ], md=3)
    ], className="mb-3")


def create_monte_carlo_simulator():
    """
    Create complete Monte Carlo simulator dashboard
    
    Components:
    - Parameter inputs (n_simulations, initial_equity, risk_per_trade)
    - Run simulation button
    - Summary cards
    - Equity curve fan chart
    - Drawdown distribution
    - Kelly Criterion display
    - Risk of ruin gauge
    - Risk comparison table
    - Insights panel
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-dice-5 me-2"),
                "Monte Carlo Simulation"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Info Alert
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                html.Strong("Monte Carlo Simulation: "),
                "Run thousands of 'what-if' scenarios to understand your risk of ruin and optimal position sizing. ",
                "This helps you determine how much to risk per trade for sustainable long-term growth."
            ], color="info", className="mb-3"),
            
            # Parameter Inputs
            dbc.Row([
                dbc.Col([
                    html.Label("Jumlah Simulasi", className="fw-bold"),
                    dbc.Input(
                        id='mc-n-simulations-input',
                        type='number',
                        min=100,
                        max=10000,
                        step=100,
                        value=1000,
                        placeholder='1000'
                    ),
                    html.Small("Recommended: 1000-5000", className="text-muted")
                ], md=3),
                dbc.Col([
                    html.Label("Ekuitas Awal ($)", className="fw-bold"),
                    dbc.Input(
                        id='mc-initial-equity-input',
                        type='number',
                        min=100,
                        step=100,
                        value=10000,
                        placeholder='10000'
                    ),
                    html.Small("Your starting capital", className="text-muted")
                ], md=3),
                dbc.Col([
                    html.Label("Risk Per Trade (%)", className="fw-bold"),
                    html.Div([
                        dcc.Slider(
                            id='mc-risk-per-trade-slider',
                            min=0.1,
                            max=5.0,
                            step=0.1,
                            value=1.0,
                            marks={
                                0.5: '0.5%',
                                1.0: '1%',
                                2.0: '2%',
                                3.0: '3%',
                                5.0: '5%'
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(id="mc-risk-display", className="text-center fw-bold mt-1")
                    ])
                ], md=4),
                dbc.Col([
                    html.Label("Actions", className="fw-bold"),
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-2"), "Jalankan Simulasi"],
                            id="mc-run-simulation-btn",
                            color="success",
                            size="lg"
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-clockwise me-2")],
                            id="mc-reset-btn",
                            color="secondary",
                            outline=True,
                            size="lg"
                        )
                    ], className="w-100")
                ], md=2)
            ], className="mb-4"),
            
            # Summary Cards
            html.Div(id="mc-summary-cards-container"),
            
            # Main Charts Row
            dbc.Row([
                # Equity Fan Chart
                dbc.Col([
                    html.H6("Equity Curve Fan Chart", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-mc-fan-chart",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='mc-equity-fan-chart',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '450px'}
                            )
                        ]
                    )
                ], md=8),
                
                # Kelly Criterion and Risk of Ruin
                dbc.Col([
                    # Kelly Criterion Card
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="bi bi-calculator me-2"),
                                "Kelly Criterion"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Div(id="mc-kelly-criterion-display")
                        ])
                    ], className="mb-3"),
                    
                    # Risk of Ruin Gauge
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="bi bi-exclamation-triangle me-2"),
                                "Risk of Ruin"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-mc-ruin-gauge",
                                type="default",
                                children=[
                                    dcc.Graph(
                                        id='mc-risk-of-ruin-gauge',
                                        config={'displayModeBar': False, 'displaylogo': False},
                                        style={'height': '250px'}
                                    )
                                ]
                            ),
                            html.Div(id="mc-ruin-warning-message")
                        ])
                    ])
                ], md=4)
            ], className="mb-3"),
            
            # Drawdown and Risk Comparison Row
            dbc.Row([
                # Drawdown Distribution
                dbc.Col([
                    html.H6("Maximum Drawdown Distribution", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-mc-drawdown",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='mc-drawdown-histogram',
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '350px'}
                            )
                        ]
                    )
                ], md=6),
                
                # Risk Comparison Table
                dbc.Col([
                    html.H6("Risk Scenario Comparison", className="fw-bold mb-2"),
                    dcc.Loading(
                        id="loading-mc-risk-comparison",
                        type="default",
                        children=[
                            html.Div(id="mc-risk-comparison-table-container")
                        ]
                    )
                ], md=6)
            ], className="mb-3"),
            
            # Insights Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.I(className="bi bi-lightbulb me-2"),
                                "Insights & Recommendations"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Div(id="mc-insights-panel")
                        ])
                    ])
                ])
            ])
        ])
    ], className="mb-4")
