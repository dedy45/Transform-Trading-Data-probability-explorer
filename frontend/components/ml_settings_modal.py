"""
ML Settings Modal Component

Provides a comprehensive settings interface for ML Prediction Engine with 4 tabs:
1. Features - Select/deselect features (minimum 5 required) with Select All/Unselect All buttons
2. Model Hyperparameters - Adjust LightGBM parameters
3. Thresholds - Adjust quality categorization thresholds
4. Display - Adjust visualization settings
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import yaml
import os


def load_config():
    """Load current configuration from YAML file"""
    config_path = 'config/ml_prediction_config.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def create_features_tab_content(config):
    """Create Features tab content"""
    if not config:
        return html.Div("Error loading configuration", className="text-danger")
    
    selected_features = config.get('features', {}).get('selected', [])
    
    # Common trading features that could be selected
    all_possible_features = [
        'trend_strength_tf', 'swing_position', 'volatility_regime',
        'support_distance', 'momentum_score', 'time_of_day',
        'spread_ratio', 'volume_profile', 'atr_ratio', 'rsi_value',
        'macd_signal', 'bb_position', 'ema_distance', 'volume_ratio',
        'price_action_score'
    ]
    
    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "Select at least 5 features for model training. More features may improve accuracy but increase training time."
        ], color="info", className="mb-3"),
        
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6("Feature Selection", className="mb-3"),
                ], md=6),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="bi bi-check-all me-1"),
                            "Select All"
                        ], id="ml-settings-select-all-btn", color="primary", size="sm", outline=True),
                        dbc.Button([
                            html.I(className="bi bi-x-circle me-1"),
                            "Unselect All"
                        ], id="ml-settings-unselect-all-btn", color="secondary", size="sm", outline=True),
                    ], className="float-end")
                ], md=6)
            ]),
            html.P(f"Selected: {len(selected_features)} features", 
                   className="text-muted mb-3", id="ml-settings-feature-count"),
            
            dbc.Checklist(
                id='ml-settings-feature-checklist',
                options=[
                    {
                        'label': html.Div([
                            html.Span(feature.replace('_', ' ').title(), className="me-2"),
                            html.Small(f"({feature})", className="text-muted")
                        ]),
                        'value': feature
                    }
                    for feature in all_possible_features
                ],
                value=selected_features,
                inline=False,
                className="mb-3"
            ),
        ]),
        
        html.Hr(),
        
        html.Div([
            html.H6("Feature Engineering Options", className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Scaling Method", className="fw-bold"),
                    dcc.Dropdown(
                        id='ml-settings-scaling',
                        options=[
                            {'label': 'Standard Scaling', 'value': 'standard'},
                            {'label': 'Min-Max Scaling', 'value': 'minmax'},
                            {'label': 'Robust Scaling', 'value': 'robust'},
                            {'label': 'No Scaling', 'value': 'none'}
                        ],
                        value=config.get('features', {}).get('scaling', 'standard'),
                        clearable=False
                    ),
                ], md=6),
                
                dbc.Col([
                    html.Label("Missing Value Handling", className="fw-bold"),
                    dcc.Dropdown(
                        id='ml-settings-missing',
                        options=[
                            {'label': 'Median Imputation', 'value': 'median'},
                            {'label': 'Mean Imputation', 'value': 'mean'},
                            {'label': 'Drop Rows', 'value': 'drop'},
                            {'label': 'Forward Fill', 'value': 'forward_fill'}
                        ],
                        value=config.get('features', {}).get('handle_missing', 'median'),
                        clearable=False
                    ),
                ], md=6),
            ])
        ])
    ])


def create_hyperparameters_tab_content(config):
    """Create Model Hyperparameters tab content"""
    if not config:
        return html.Div("Error loading configuration", className="text-danger")
    
    classifier_params = config.get('model_hyperparameters', {}).get('classifier', {})
    quantile_params = config.get('model_hyperparameters', {}).get('quantile', {})
    
    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "Adjust model hyperparameters. Higher values may improve accuracy but increase training time and risk overfitting."
        ], color="info", className="mb-3"),
        
        # Classifier Parameters
        html.Div([
            html.H6([
                html.I(className="bi bi-diagram-3 me-2"),
                "Classifier Parameters (LightGBM)"
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Number of Estimators", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-classifier-n-estimators',
                        type='number',
                        value=classifier_params.get('n_estimators', 100),
                        min=10,
                        max=500,
                        step=10
                    ),
                    html.Small("Number of boosting iterations (10-500)", className="text-muted")
                ], md=4),
                
                dbc.Col([
                    html.Label("Learning Rate", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-classifier-learning-rate',
                        type='number',
                        value=classifier_params.get('learning_rate', 0.05),
                        min=0.001,
                        max=0.3,
                        step=0.005
                    ),
                    html.Small("Step size for each iteration (0.001-0.3)", className="text-muted")
                ], md=4),
                
                dbc.Col([
                    html.Label("Max Depth", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-classifier-max-depth',
                        type='number',
                        value=classifier_params.get('max_depth', 5),
                        min=2,
                        max=15,
                        step=1
                    ),
                    html.Small("Maximum tree depth (2-15)", className="text-muted")
                ], md=4),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Min Child Samples", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-classifier-min-child-samples',
                        type='number',
                        value=classifier_params.get('min_child_samples', 20),
                        min=5,
                        max=100,
                        step=5
                    ),
                    html.Small("Minimum samples per leaf (5-100)", className="text-muted")
                ], md=4),
                
                dbc.Col([
                    html.Label("Subsample", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-classifier-subsample',
                        type='number',
                        value=classifier_params.get('subsample', 0.8),
                        min=0.5,
                        max=1.0,
                        step=0.05
                    ),
                    html.Small("Fraction of samples for training (0.5-1.0)", className="text-muted")
                ], md=4),
                
                dbc.Col([
                    html.Label("Column Sample by Tree", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-classifier-colsample',
                        type='number',
                        value=classifier_params.get('colsample_bytree', 0.8),
                        min=0.5,
                        max=1.0,
                        step=0.05
                    ),
                    html.Small("Fraction of features per tree (0.5-1.0)", className="text-muted")
                ], md=4),
            ])
        ]),
        
        html.Hr(className="my-4"),
        
        # Quantile Regressor Parameters
        html.Div([
            html.H6([
                html.I(className="bi bi-bar-chart me-2"),
                "Quantile Regressor Parameters"
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Number of Estimators", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-quantile-n-estimators',
                        type='number',
                        value=quantile_params.get('n_estimators', 100),
                        min=10,
                        max=500,
                        step=10
                    ),
                    html.Small("Number of boosting iterations (10-500)", className="text-muted")
                ], md=4),
                
                dbc.Col([
                    html.Label("Learning Rate", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-quantile-learning-rate',
                        type='number',
                        value=quantile_params.get('learning_rate', 0.05),
                        min=0.001,
                        max=0.3,
                        step=0.005
                    ),
                    html.Small("Step size for each iteration (0.001-0.3)", className="text-muted")
                ], md=4),
                
                dbc.Col([
                    html.Label("Max Depth", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-quantile-max-depth',
                        type='number',
                        value=quantile_params.get('max_depth', 5),
                        min=2,
                        max=15,
                        step=1
                    ),
                    html.Small("Maximum tree depth (2-15)", className="text-muted")
                ], md=4),
            ])
        ])
    ])


def create_thresholds_tab_content(config):
    """Create Thresholds tab content"""
    if not config:
        return html.Div("Error loading configuration", className="text-danger")
    
    thresholds = config.get('thresholds', {})
    
    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "Adjust quality categorization thresholds. Higher thresholds mean stricter quality requirements."
        ], color="info", className="mb-3"),
        
        # A+ Category
        html.Div([
            html.H6([
                html.Span("A+ Category", className="me-2"),
                dbc.Badge("Excellent", color="success", className="me-2"),
                html.Small("(Dark Green)", className="text-muted")
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Minimum Probability Win", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-threshold-aplus-prob',
                        type='number',
                        value=thresholds.get('quality_A_plus', {}).get('prob_win_min', 0.65),
                        min=0.5,
                        max=1.0,
                        step=0.01
                    ),
                    html.Small("Minimum win probability for A+ (0.5-1.0)", className="text-muted")
                ], md=6),
                
                dbc.Col([
                    html.Label("Minimum Expected R", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-threshold-aplus-r',
                        type='number',
                        value=thresholds.get('quality_A_plus', {}).get('R_P50_min', 1.5),
                        min=0.0,
                        max=5.0,
                        step=0.1
                    ),
                    html.Small("Minimum R_P50 for A+ (0.0-5.0)", className="text-muted")
                ], md=6),
            ])
        ], className="mb-4"),
        
        # A Category
        html.Div([
            html.H6([
                html.Span("A Category", className="me-2"),
                dbc.Badge("Good", color="success", className="me-2"),
                html.Small("(Green)", className="text-muted")
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Minimum Probability Win", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-threshold-a-prob',
                        type='number',
                        value=thresholds.get('quality_A', {}).get('prob_win_min', 0.55),
                        min=0.5,
                        max=1.0,
                        step=0.01
                    ),
                    html.Small("Minimum win probability for A (0.5-1.0)", className="text-muted")
                ], md=6),
                
                dbc.Col([
                    html.Label("Minimum Expected R", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-threshold-a-r',
                        type='number',
                        value=thresholds.get('quality_A', {}).get('R_P50_min', 1.0),
                        min=0.0,
                        max=5.0,
                        step=0.1
                    ),
                    html.Small("Minimum R_P50 for A (0.0-5.0)", className="text-muted")
                ], md=6),
            ])
        ], className="mb-4"),
        
        # B Category
        html.Div([
            html.H6([
                html.Span("B Category", className="me-2"),
                dbc.Badge("Fair", color="warning", className="me-2"),
                html.Small("(Yellow)", className="text-muted")
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Minimum Probability Win", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-threshold-b-prob',
                        type='number',
                        value=thresholds.get('quality_B', {}).get('prob_win_min', 0.45),
                        min=0.0,
                        max=1.0,
                        step=0.01
                    ),
                    html.Small("Minimum win probability for B (0.0-1.0)", className="text-muted")
                ], md=6),
                
                dbc.Col([
                    html.Label("Minimum Expected R", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-threshold-b-r',
                        type='number',
                        value=thresholds.get('quality_B', {}).get('R_P50_min', 0.5),
                        min=0.0,
                        max=5.0,
                        step=0.1
                    ),
                    html.Small("Minimum R_P50 for B (0.0-5.0)", className="text-muted")
                ], md=6),
            ])
        ], className="mb-4"),
        
        html.Hr(),
        
        # Recommendation Settings
        html.Div([
            html.H6("Recommendation Settings", className="mb-3"),
            
            html.Label("Minimum Quality for TRADE Recommendation", className="fw-bold"),
            dcc.Dropdown(
                id='ml-settings-trade-quality-min',
                options=[
                    {'label': 'A+ Only', 'value': 'A+'},
                    {'label': 'A or Better (A+, A)', 'value': 'A'},
                    {'label': 'B or Better (A+, A, B)', 'value': 'B'},
                ],
                value=thresholds.get('trade_quality_min', 'A'),
                clearable=False
            ),
            html.Small("Setups below this quality will get SKIP recommendation", className="text-muted")
        ])
    ])


def create_display_tab_content(config):
    """Create Display tab content"""
    if not config:
        return html.Div("Error loading configuration", className="text-danger")
    
    display = config.get('display', {})
    color_scheme = display.get('color_scheme', {})
    
    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            "Customize visualization settings and color schemes for the ML Prediction Engine."
        ], color="info", className="mb-3"),
        
        # Visualization Settings
        html.Div([
            html.H6("Visualization Settings", className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Reliability Diagram Bins", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-n-bins',
                        type='number',
                        value=display.get('n_bins_reliability', 10),
                        min=5,
                        max=20,
                        step=1
                    ),
                    html.Small("Number of bins for reliability diagram (5-20)", className="text-muted")
                ], md=6),
                
                dbc.Col([
                    html.Label("Top Features to Display", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-n-top-features',
                        type='number',
                        value=display.get('n_top_features', 10),
                        min=5,
                        max=20,
                        step=1
                    ),
                    html.Small("Number of top features in importance chart (5-20)", className="text-muted")
                ], md=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Chart Height (pixels)", className="fw-bold"),
                    dbc.Input(
                        id='ml-settings-chart-height',
                        type='number',
                        value=display.get('chart_height', 400),
                        min=300,
                        max=800,
                        step=50
                    ),
                    html.Small("Default height for charts (300-800)", className="text-muted")
                ], md=6),
                
                dbc.Col([
                    html.Label("Chart Template", className="fw-bold"),
                    dcc.Dropdown(
                        id='ml-settings-chart-template',
                        options=[
                            {'label': 'Plotly White', 'value': 'plotly_white'},
                            {'label': 'Plotly Dark', 'value': 'plotly_dark'},
                            {'label': 'Simple White', 'value': 'simple_white'},
                            {'label': 'Seaborn', 'value': 'seaborn'},
                        ],
                        value=display.get('chart_template', 'plotly_white'),
                        clearable=False
                    ),
                    html.Small("Visual theme for charts", className="text-muted")
                ], md=6),
            ])
        ]),
        
        html.Hr(className="my-4"),
        
        # Color Scheme
        html.Div([
            html.H6("Quality Category Colors", className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("A+ Color (Excellent)", className="fw-bold"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='ml-settings-color-aplus',
                            type='text',
                            value=color_scheme.get('A_plus', '#006400'),
                            placeholder='#006400'
                        ),
                        dbc.InputGroupText(
                            html.Div(style={
                                'width': '30px',
                                'height': '30px',
                                'backgroundColor': color_scheme.get('A_plus', '#006400'),
                                'border': '1px solid #ccc'
                            })
                        )
                    ]),
                    html.Small("Hex color code for A+ category", className="text-muted")
                ], md=6),
                
                dbc.Col([
                    html.Label("A Color (Good)", className="fw-bold"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='ml-settings-color-a',
                            type='text',
                            value=color_scheme.get('A', '#32CD32'),
                            placeholder='#32CD32'
                        ),
                        dbc.InputGroupText(
                            html.Div(style={
                                'width': '30px',
                                'height': '30px',
                                'backgroundColor': color_scheme.get('A', '#32CD32'),
                                'border': '1px solid #ccc'
                            })
                        )
                    ]),
                    html.Small("Hex color code for A category", className="text-muted")
                ], md=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("B Color (Fair)", className="fw-bold"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='ml-settings-color-b',
                            type='text',
                            value=color_scheme.get('B', '#FFD700'),
                            placeholder='#FFD700'
                        ),
                        dbc.InputGroupText(
                            html.Div(style={
                                'width': '30px',
                                'height': '30px',
                                'backgroundColor': color_scheme.get('B', '#FFD700'),
                                'border': '1px solid #ccc'
                            })
                        )
                    ]),
                    html.Small("Hex color code for B category", className="text-muted")
                ], md=6),
                
                dbc.Col([
                    html.Label("C Color (Poor)", className="fw-bold"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='ml-settings-color-c',
                            type='text',
                            value=color_scheme.get('C', '#DC143C'),
                            placeholder='#DC143C'
                        ),
                        dbc.InputGroupText(
                            html.Div(style={
                                'width': '30px',
                                'height': '30px',
                                'backgroundColor': color_scheme.get('C', '#DC143C'),
                                'border': '1px solid #ccc'
                            })
                        )
                    ]),
                    html.Small("Hex color code for C category", className="text-muted")
                ], md=6),
            ])
        ])
    ])


def get_settings_content(active_tab):
    """
    Get content for the active settings tab
    
    Parameters
    ----------
    active_tab : str
        Active tab ID (settings-features, settings-hyperparams, settings-thresholds, settings-display)
    
    Returns
    -------
    dash component
        Content for the active tab
    """
    config = load_config()
    
    if active_tab == 'settings-features':
        return create_features_tab_content(config)
    elif active_tab == 'settings-hyperparams':
        return create_hyperparameters_tab_content(config)
    elif active_tab == 'settings-thresholds':
        return create_thresholds_tab_content(config)
    elif active_tab == 'settings-display':
        return create_display_tab_content(config)
    else:
        return html.Div("Select a tab to view settings", className="text-muted")


def save_settings_to_config(settings_dict):
    """
    Save settings to ml_prediction_config.yaml
    
    Parameters
    ----------
    settings_dict : dict
        Dictionary containing all settings to save
    
    Returns
    -------
    tuple
        (success: bool, message: str)
    """
    config_path = 'config/ml_prediction_config.yaml'
    
    try:
        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config with new settings
        if 'features' in settings_dict:
            config['features'] = settings_dict['features']
        
        if 'model_hyperparameters' in settings_dict:
            config['model_hyperparameters'] = settings_dict['model_hyperparameters']
        
        if 'thresholds' in settings_dict:
            config['thresholds'] = settings_dict['thresholds']
        
        if 'display' in settings_dict:
            config['display'] = settings_dict['display']
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True, "Settings saved successfully!"
    
    except Exception as e:
        return False, f"Error saving settings: {str(e)}"
