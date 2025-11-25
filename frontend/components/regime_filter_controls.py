"""
Regime Filter Controls Component
Interactive controls for filtering and analyzing specific regimes
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_regime_filter_controls(available_regimes=None):
    """
    Create filter controls for regime selection and analysis settings
    
    Parameters:
    -----------
    available_regimes : list, optional
        List of available regime values from the dataset
        If None, shows placeholder
    
    Returns:
    --------
    dash_bootstrap_components.Card
        Card containing all regime filter controls
    
    Controls:
    ---------
    - Regime column selector (dropdown)
    - Target variable selector (dropdown)
    - Confidence level slider
    - Minimum samples slider
    - Regime multi-select (checklist)
    - Calculate button
    - Export button
    - Help button
    """
    if available_regimes is None:
        regime_options = []
        regime_value = []
    else:
        regime_options = [{'label': str(regime), 'value': regime} for regime in available_regimes]
        regime_value = available_regimes  # Select all by default
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Regime Analysis Settings", className="mb-0")),
        dbc.CardBody([
            # Row 1: Main settings
            dbc.Row([
                dbc.Col([
                    html.Label("Regime Column", className="fw-bold"),
                    dcc.Dropdown(
                        id='regime-column-dropdown',
                        options=[
                            {'label': 'Trend Regime', 'value': 'trend_regime'},
                            {'label': 'Volatility Regime', 'value': 'volatility_regime'},
                            {'label': 'Risk Regime Global', 'value': 'risk_regime_global'},
                            {'label': 'Session', 'value': 'session'},
                            {'label': 'Custom Regime', 'value': 'custom_regime'}
                        ],
                        value='trend_regime',
                        clearable=False,
                        placeholder="Select regime column"
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Target Variable", className="fw-bold"),
                    dcc.Dropdown(
                        id='regime-target-variable-dropdown',
                        options=[
                            {'label': 'Trade Success (Win/Loss)', 'value': 'trade_success'},
                            {'label': 'Hit 1R', 'value': 'y_hit_1R'},
                            {'label': 'Hit 2R', 'value': 'y_hit_2R'},
                        ],
                        value='trade_success',
                        clearable=False
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Confidence Level (%)", className="fw-bold"),
                    dcc.Slider(
                        id='regime-confidence-level-slider',
                        min=80,
                        max=99,
                        step=1,
                        value=95,
                        marks={80: '80%', 85: '85%', 90: '90%', 95: '95%', 99: '99%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Min Samples per Regime", className="fw-bold"),
                    dcc.Slider(
                        id='regime-min-samples-slider',
                        min=3,
                        max=30,
                        step=1,
                        value=5,
                        marks={3: '3', 10: '10', 20: '20', 30: '30'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=3)
            ], className="mb-3"),
            
            # Row 2: Regime selection
            dbc.Row([
                dbc.Col([
                    html.Label("Select Regimes to Analyze", className="fw-bold"),
                    html.Div([
                        dcc.Checklist(
                            id='regime-selection-checklist',
                            options=regime_options if regime_options else [],
                            value=regime_value if regime_value else [],
                            inline=True,
                            labelStyle={'display': 'inline-block', 'marginRight': '15px'},
                            inputStyle={'marginRight': '5px'}
                        ),
                        html.P(
                            "Load data to see available regimes" if not regime_options else "",
                            className="text-muted small mb-0"
                        )
                    ], style={'maxHeight': '100px', 'overflowY': 'auto', 'padding': '10px', 
                             'border': '1px solid #ddd', 'borderRadius': '4px', 
                             'backgroundColor': '#f8f9fa'})
                ], md=12)
            ], className="mb-3"),
            
            # Row 3: Quick select buttons
            dbc.Row([
                dbc.Col([
                    html.Label("Quick Select:", className="fw-bold me-2"),
                    dbc.ButtonGroup([
                        dbc.Button("All", id="regime-select-all-btn", size="sm", color="secondary", outline=True),
                        dbc.Button("None", id="regime-select-none-btn", size="sm", color="secondary", outline=True),
                        dbc.Button("Invert", id="regime-select-invert-btn", size="sm", color="secondary", outline=True)
                    ], size="sm")
                ], md=12)
            ], className="mb-3"),
            
            # Row 4: Action buttons
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="bi bi-calculator me-2"), "Calculate Regime Analysis"],
                            id="regime-calculate-btn",
                            color="primary",
                            size="lg"
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-download me-2"), "Export Results"],
                            id="regime-export-btn",
                            color="success",
                            size="lg",
                            outline=True
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-info-circle me-2"), "Regime Guide"],
                            id="regime-help-btn",
                            color="info",
                            size="lg",
                            outline=True
                        )
                    ], className="w-100")
                ])
            ])
        ])
    ], className="mb-3")


def create_regime_summary_cards(regime_comparison_df):
    """
    Create summary cards showing key regime statistics
    
    Parameters:
    -----------
    regime_comparison_df : pd.DataFrame
        Results from create_regime_comparison_table
    
    Returns:
    --------
    dash_bootstrap_components.Row
        Row containing 4 summary cards
    
    Cards:
    ------
    1. Best Regime - Highest win rate
    2. Worst Regime - Lowest win rate
    3. Highest Mean R - Best average R-multiple
    4. Total Regimes - Number of unique regimes analyzed
    """
    if regime_comparison_df is None or regime_comparison_df.empty:
        return create_empty_regime_summary_cards()
    
    df = regime_comparison_df.copy()
    
    # Find best and worst regimes
    best_regime_idx = df['win_rate'].idxmax()
    worst_regime_idx = df['win_rate'].idxmin()
    highest_r_idx = df['mean_r'].idxmax()
    
    best_regime = df.loc[best_regime_idx]
    worst_regime = df.loc[worst_regime_idx]
    highest_r_regime = df.loc[highest_r_idx]
    
    cards = [
        {
            'title': 'Best Regime',
            'value': str(best_regime['regime']),
            'subtitle': f"Win Rate: {best_regime['win_rate']:.1%}",
            'description': f"Mean R: {best_regime['mean_r']:.2f} | Trades: {int(best_regime['n_trades'])}",
            'color': 'success',
            'icon': 'bi-trophy-fill'
        },
        {
            'title': 'Worst Regime',
            'value': str(worst_regime['regime']),
            'subtitle': f"Win Rate: {worst_regime['win_rate']:.1%}",
            'description': f"Mean R: {worst_regime['mean_r']:.2f} | Trades: {int(worst_regime['n_trades'])}",
            'color': 'danger',
            'icon': 'bi-exclamation-triangle-fill'
        },
        {
            'title': 'Highest Mean R',
            'value': str(highest_r_regime['regime']),
            'subtitle': f"Mean R: {highest_r_regime['mean_r']:.2f}",
            'description': f"Win Rate: {highest_r_regime['win_rate']:.1%} | Trades: {int(highest_r_regime['n_trades'])}",
            'color': 'info',
            'icon': 'bi-graph-up-arrow'
        },
        {
            'title': 'Total Regimes',
            'value': str(len(df)),
            'subtitle': f"Reliable: {df['reliable'].sum()}",
            'description': f"Total Trades: {int(df['n_trades'].sum())}",
            'color': 'primary',
            'icon': 'bi-grid-3x3-gap-fill'
        }
    ]
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className=f"{card['icon']} me-2", style={'fontSize': '1.5rem'}),
                        html.Span(card['title'], className="fw-bold")
                    ], className="d-flex align-items-center mb-2"),
                    html.H3(card['value'], className=f"text-{card['color']} mb-1"),
                    html.P(card['subtitle'], className="text-muted small mb-1"),
                    html.P(card['description'], className="text-muted small mb-0")
                ])
            ], className="h-100", color=card['color'], outline=True)
        ], md=3)
        for card in cards
    ], className="mb-3")


def create_empty_regime_summary_cards():
    """Create empty summary cards placeholder"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("No Data", className="text-muted text-center"),
                    html.P("Calculate regime analysis", className="text-center small mb-0")
                ])
            ], className="text-center")
        ], md=3)
        for _ in range(4)
    ], className="mb-3")
