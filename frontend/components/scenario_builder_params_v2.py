"""
Scenario Builder Parameter Panels V2
Dynamic parameter inputs with pattern matching IDs
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_position_sizing_params():
    """Create parameter inputs for position sizing scenario"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Risk Per Trade (%)", className="fw-bold small"),
                dbc.Input(
                    id={'type': 'scenario-param', 'name': 'risk_percent'},
                    type='number',
                    min=0.1,
                    max=5.0,
                    step=0.1,
                    value=1.0,
                    size='sm'
                ),
                html.Small("Range: 0.1% - 5.0%", className="text-muted")
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Max Position Size (optional)", className="fw-bold small"),
                dbc.Input(
                    id={'type': 'scenario-param', 'name': 'max_position'},
                    type='number',
                    min=0.01,
                    step=0.01,
                    placeholder='Leave empty for no limit',
                    size='sm'
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Fixed Lot Size (optional)", className="fw-bold small"),
                dbc.Input(
                    id={'type': 'scenario-param', 'name': 'fixed_lot'},
                    type='number',
                    min=0.01,
                    step=0.01,
                    placeholder='Leave empty to use risk %',
                    size='sm'
                )
            ])
        ])
    ])


def create_sl_tp_params():
    """Create parameter inputs for SL/TP scenario"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Stop Loss Multiplier", className="fw-bold small"),
                dcc.Slider(
                    id={'type': 'scenario-param', 'name': 'sl_multiplier'},
                    min=0.5,
                    max=3.0,
                    step=0.1,
                    value=1.0,
                    marks={0.5: '0.5x', 1.0: '1.0x', 1.5: '1.5x', 2.0: '2.0x', 2.5: '2.5x', 3.0: '3.0x'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Tighter SL < 1.0 < Wider SL", className="text-muted")
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Take Profit Multiplier", className="fw-bold small"),
                dcc.Slider(
                    id={'type': 'scenario-param', 'name': 'tp_multiplier'},
                    min=0.5,
                    max=5.0,
                    step=0.1,
                    value=1.0,
                    marks={0.5: '0.5x', 1.0: '1.0x', 2.0: '2.0x', 3.0: '3.0x', 4.0: '4.0x', 5.0: '5.0x'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Closer TP < 1.0 < Further TP", className="text-muted")
            ])
        ])
    ])


def create_filter_params():
    """Create parameter inputs for trade filtering scenario"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Min Probability", className="fw-bold small"),
                dcc.Slider(
                    id={'type': 'scenario-param', 'name': 'min_probability'},
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.5,
                    marks={0: '0%', 0.25: '25%', 0.5: '50%', 0.75: '75%', 1.0: '100%'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Min Composite Score", className="fw-bold small"),
                dcc.Slider(
                    id={'type': 'scenario-param', 'name': 'min_score'},
                    min=0,
                    max=100,
                    step=5,
                    value=50,
                    marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ], className="mb-2"),
    ])


def create_time_params():
    """Create parameter inputs for time restrictions scenario"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Trading Hours", className="fw-bold small"),
                dcc.RangeSlider(
                    id={'type': 'scenario-param', 'name': 'trading_hours'},
                    min=0,
                    max=23,
                    step=1,
                    value=[0, 23],
                    marks={0: '0h', 6: '6h', 12: '12h', 18: '18h', 23: '23h'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ], className="mb-3"),
    ])


def create_market_condition_params():
    """Create parameter inputs for market condition scenario"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Trend Regime", className="fw-bold small"),
                dbc.Checklist(
                    id={'type': 'scenario-param', 'name': 'trend_regime'},
                    options=[
                        {'label': ' Ranging', 'value': 0},
                        {'label': ' Trending', 'value': 1},
                    ],
                    value=[0, 1],
                    inline=True
                )
            ])
        ], className="mb-2"),
    ])


def create_money_management_params():
    """Create parameter inputs for money management scenario"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    id={'type': 'scenario-param', 'name': 'compounding'},
                    options=[{'label': ' Enable Compounding', 'value': True}],
                    value=[],
                    switch=True
                )
            ])
        ], className="mb-2"),
    ])


def create_ml_prediction_params():
    """Create parameter inputs for ML prediction scenario"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Filter by Quality", className="fw-bold small"),
                dbc.Checklist(
                    id={'type': 'scenario-param', 'name': 'ml_quality_filter'},
                    options=[
                        {'label': ' A+ (Excellent)', 'value': 'A+'},
                        {'label': ' A (Good)', 'value': 'A'},
                        {'label': ' B (Fair)', 'value': 'B'},
                        {'label': ' C (Poor)', 'value': 'C'},
                    ],
                    value=['A+', 'A'],
                    inline=False
                ),
                html.Small("Select quality levels to include", className="text-muted d-block mb-2")
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Min Win Probability", className="fw-bold small"),
                dcc.Slider(
                    id={'type': 'scenario-param', 'name': 'ml_min_prob'},
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.55,
                    marks={0: '0%', 0.25: '25%', 0.5: '50%', 0.75: '75%', 1.0: '100%'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Minimum calibrated win probability", className="text-muted")
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    id={'type': 'scenario-param', 'name': 'ml_use_recommendation'},
                    options=[{'label': ' Only TRADE recommendations', 'value': True}],
                    value=[True],
                    switch=True
                ),
                html.Small("Filter by ML recommendation (TRADE/SKIP)", className="text-muted")
            ])
        ], className="mb-2"),
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "This scenario requires ML predictions to be available. ",
            "Run predictions in the ML Prediction Engine first."
        ], color="info", className="mt-3 small")
    ])


def get_scenario_params_panel(scenario_type):
    """
    Get the appropriate parameter panel based on scenario type
    
    Args:
        scenario_type: Type of scenario
    
    Returns:
        Dash component with parameter inputs
    """
    if scenario_type == 'position_sizing':
        return create_position_sizing_params()
    elif scenario_type == 'sl_tp':
        return create_sl_tp_params()
    elif scenario_type == 'filter':
        return create_filter_params()
    elif scenario_type == 'time':
        return create_time_params()
    elif scenario_type == 'market_condition':
        return create_market_condition_params()
    elif scenario_type == 'money_management':
        return create_money_management_params()
    elif scenario_type == 'ml_prediction':
        return create_ml_prediction_params()
    else:
        return html.Div("Select a scenario type", className="text-muted")
