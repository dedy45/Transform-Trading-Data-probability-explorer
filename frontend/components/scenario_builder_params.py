"""
Scenario Builder Parameter Panels
Dynamic parameter inputs for different scenario types
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_position_sizing_params():
    """
    Create parameter inputs for position sizing scenario
    
    Parameters:
    - Risk percent per trade (0.1 - 5.0%)
    - Max position size
    - Fixed lot size (optional)
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Risk Per Trade (%)", className="fw-bold small"),
                dbc.Input(
                    id='param-risk-percent',
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
                    id='param-max-position',
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
                    id='param-fixed-lot',
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
    """
    Create parameter inputs for SL/TP scenario
    
    Parameters:
    - SL multiplier (0.5 - 3.0)
    - TP multiplier (0.5 - 5.0)
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Stop Loss Multiplier", className="fw-bold small"),
                dcc.Slider(
                    id='param-sl-multiplier',
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
                    id='param-tp-multiplier',
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
    """
    Create parameter inputs for trade filtering scenario
    
    Parameters:
    - Min probability threshold
    - Min composite score
    - Session filter
    - Trend alignment
    - Volatility regime
    - Min trend strength
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Min Probability", className="fw-bold small"),
                dcc.Slider(
                    id='param-min-probability',
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
                    id='param-min-score',
                    min=0,
                    max=100,
                    step=5,
                    value=50,
                    marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Sessions", className="fw-bold small"),
                dbc.Checklist(
                    id='param-sessions',
                    options=[
                        {'label': ' ASIA', 'value': 'ASIA'},
                        {'label': ' EUROPE', 'value': 'EUROPE'},
                        {'label': ' US', 'value': 'US'},
                    ],
                    value=['ASIA', 'EUROPE', 'US'],
                    inline=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    id='param-trend-alignment',
                    options=[{'label': ' Require Trend Alignment', 'value': True}],
                    value=[],
                    switch=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Volatility Regime", className="fw-bold small"),
                dbc.Checklist(
                    id='param-volatility-regime',
                    options=[
                        {'label': ' Low', 'value': 0},
                        {'label': ' Med', 'value': 1},
                        {'label': ' High', 'value': 2},
                    ],
                    value=[0, 1, 2],
                    inline=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Min Trend Strength", className="fw-bold small"),
                dcc.Slider(
                    id='param-min-trend-strength',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0,
                    marks={0: '0', 0.5: '0.5', 1.0: '1.0'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
    ])


def create_time_params():
    """
    Create parameter inputs for time restrictions scenario
    
    Parameters:
    - Trading hours (start, end)
    - Days of week
    - Sessions only
    - Avoid first/last hour
    - News blackout minutes
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Trading Hours", className="fw-bold small"),
                dcc.RangeSlider(
                    id='param-trading-hours',
                    min=0,
                    max=23,
                    step=1,
                    value=[0, 23],
                    marks={0: '0h', 6: '6h', 12: '12h', 18: '18h', 23: '23h'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Days of Week", className="fw-bold small"),
                dbc.Checklist(
                    id='param-days-of-week',
                    options=[
                        {'label': ' Mon', 'value': 0},
                        {'label': ' Tue', 'value': 1},
                        {'label': ' Wed', 'value': 2},
                        {'label': ' Thu', 'value': 3},
                        {'label': ' Fri', 'value': 4},
                        {'label': ' Sat', 'value': 5},
                        {'label': ' Sun', 'value': 6},
                    ],
                    value=[0, 1, 2, 3, 4],
                    inline=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Sessions Only", className="fw-bold small"),
                dbc.Checklist(
                    id='param-sessions-only',
                    options=[
                        {'label': ' ASIA', 'value': 'ASIA'},
                        {'label': ' EUROPE', 'value': 'EUROPE'},
                        {'label': ' US', 'value': 'US'},
                    ],
                    value=[],
                    inline=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("News Blackout (minutes)", className="fw-bold small"),
                dbc.Input(
                    id='param-news-blackout',
                    type='number',
                    min=0,
                    max=120,
                    step=5,
                    value=0,
                    size='sm'
                ),
                html.Small("Avoid trading X minutes before/after news", className="text-muted")
            ])
        ])
    ])


def create_market_condition_params():
    """
    Create parameter inputs for market condition scenario
    
    Parameters:
    - Trend regime
    - Volatility regime
    - Risk regime
    - Entropy range
    - Hurst exponent range
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Trend Regime", className="fw-bold small"),
                dbc.Checklist(
                    id='param-trend-regime',
                    options=[
                        {'label': ' Ranging', 'value': 0},
                        {'label': ' Trending', 'value': 1},
                    ],
                    value=[0, 1],
                    inline=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Volatility Regime", className="fw-bold small"),
                dbc.Checklist(
                    id='param-vol-regime',
                    options=[
                        {'label': ' Low', 'value': 0},
                        {'label': ' Medium', 'value': 1},
                        {'label': ' High', 'value': 2},
                    ],
                    value=[0, 1, 2],
                    inline=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Risk Regime", className="fw-bold small"),
                dbc.Checklist(
                    id='param-risk-regime',
                    options=[
                        {'label': ' Risk-On', 'value': 0},
                        {'label': ' Risk-Off', 'value': 1},
                    ],
                    value=[0, 1],
                    inline=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Entropy Range", className="fw-bold small"),
                dcc.RangeSlider(
                    id='param-entropy-range',
                    min=0,
                    max=1,
                    step=0.1,
                    value=[0, 1],
                    marks={0: '0', 0.5: '0.5', 1.0: '1.0'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Hurst Exponent Range", className="fw-bold small"),
                dcc.RangeSlider(
                    id='param-hurst-range',
                    min=0,
                    max=1,
                    step=0.1,
                    value=[0, 1],
                    marks={0: '0', 0.5: '0.5', 1.0: '1.0'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
    ])


def create_money_management_params():
    """
    Create parameter inputs for money management scenario
    
    Parameters:
    - Compounding (on/off)
    - Martingale multiplier
    - Anti-martingale multiplier
    - Max consecutive losses
    - Daily profit target
    - Daily loss limit
    - Drawdown reduction percentage
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    id='param-compounding',
                    options=[{'label': ' Enable Compounding', 'value': True}],
                    value=[],
                    switch=True
                )
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Martingale Multiplier", className="fw-bold small"),
                dbc.Input(
                    id='param-martingale',
                    type='number',
                    min=1.0,
                    max=3.0,
                    step=0.1,
                    value=1.0,
                    size='sm'
                ),
                html.Small("Multiply position after loss (1.0 = no martingale)", className="text-muted")
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Anti-Martingale Multiplier", className="fw-bold small"),
                dbc.Input(
                    id='param-anti-martingale',
                    type='number',
                    min=1.0,
                    max=3.0,
                    step=0.1,
                    value=1.0,
                    size='sm'
                ),
                html.Small("Multiply position after win (1.0 = no anti-martingale)", className="text-muted")
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Max Consecutive Losses", className="fw-bold small"),
                dbc.Input(
                    id='param-max-losses',
                    type='number',
                    min=0,
                    max=10,
                    step=1,
                    value=0,
                    size='sm'
                ),
                html.Small("Pause trading after N losses (0 = no limit)", className="text-muted")
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Daily Profit Target ($)", className="fw-bold small"),
                dbc.Input(
                    id='param-daily-target',
                    type='number',
                    min=0,
                    step=50,
                    value=0,
                    size='sm'
                ),
                html.Small("Stop trading after reaching target (0 = no limit)", className="text-muted")
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Daily Loss Limit ($)", className="fw-bold small"),
                dbc.Input(
                    id='param-daily-limit',
                    type='number',
                    min=0,
                    step=50,
                    value=0,
                    size='sm'
                ),
                html.Small("Stop trading after hitting limit (0 = no limit)", className="text-muted")
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Drawdown Reduction (%)", className="fw-bold small"),
                dcc.Slider(
                    id='param-dd-reduction',
                    min=0,
                    max=50,
                    step=5,
                    value=0,
                    marks={0: '0%', 25: '25%', 50: '50%'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Reduce position size during drawdown", className="text-muted")
            ])
        ])
    ])


def get_scenario_params_panel(scenario_type):
    """
    Get the appropriate parameter panel based on scenario type
    
    Args:
        scenario_type: Type of scenario ('position_sizing', 'sl_tp', etc.)
    
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
    else:
        return html.Div("Select a scenario type", className="text-muted")
